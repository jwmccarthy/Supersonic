#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

class ReflectionVisitor : public RecursiveASTVisitor<ReflectionVisitor>
{
  private:
	ASTContext& Context;
	std::vector<const CXXRecordDecl*> ReflectedClasses;

  public:
	explicit ReflectionVisitor(ASTContext& Context) : Context(Context) {}

	bool VisitCXXRecordDecl(CXXRecordDecl* Declaration)
	{
		if (!Declaration->isCompleteDefinition())
		{
			return true;
		}

		// Check for reflect annotation
		for (const auto* Attr : Declaration->attrs())
		{
			if (auto* Annotate = dyn_cast<AnnotateAttr>(Attr))
			{
				if (Annotate->getAnnotation() == "reflect")
				{
					ReflectedClasses.push_back(Declaration);
					break;
				}
			}
		}
		return true;
	}

	void GenerateReflectionData()
	{
		if (ReflectedClasses.empty()) return;

		// Just output to current directory
		std::error_code EC;
		llvm::raw_fd_ostream OS("reflection.gen.hpp", EC);
		if (EC) return;

		OS << "#pragma once\n\n";
		OS << "#include <Reflection.hpp>\n";
		OS << "#include <array>\n\n";

		// Forward-declare structs
		for (const auto* ClassDecl : ReflectedClasses)
		{
			OS << "struct " << ClassDecl->getNameAsString() << ";\n\n";
		}

		OS << "namespace reflection {\n\n";

		for (const auto* ClassDecl : ReflectedClasses)
		{
			generateClassReflection(ClassDecl, OS);
		}

		OS << "}";
	}

  private:
	void collectFieldsRecursively(const CXXRecordDecl* ClassDecl,
	                              std::vector<const FieldDecl*>& Fields)
	{
		// First collect fields from base classes
		for (const auto& Base : ClassDecl->bases())
		{
			if (const auto* BaseClassDecl =
			        Base.getType()->getAsCXXRecordDecl())
			{
				collectFieldsRecursively(BaseClassDecl, Fields);
			}
		}

		// Then collect fields from this class
		for (const auto* Field : ClassDecl->fields())
		{
			Fields.push_back(Field);
		}
	}

	uint64_t calculateFieldOffsetInDerivedClass(const CXXRecordDecl* DerivedClass,
	                                   			const FieldDecl* Field)
	{
		const CXXRecordDecl* FieldClass =
		    cast<CXXRecordDecl>(Field->getParent());

		if (FieldClass == DerivedClass)
		{
			// Direct field - use the class layout directly
			const ASTRecordLayout& Layout =
			    Context.getASTRecordLayout(DerivedClass);
			uint64_t FieldOffsetBits =
			    Layout.getFieldOffset(Field->getFieldIndex());
			return FieldOffsetBits / 8;
		}

		// Inherited field - walk up the inheritance chain
		uint64_t TotalOffset = 0;
		const CXXRecordDecl* CurrentClass = DerivedClass;

		// Walk up until we find the class that declares the field
		while (CurrentClass != FieldClass)
		{
			const ASTRecordLayout& CurrentLayout =
			    Context.getASTRecordLayout(CurrentClass);
			bool Found = false;

			// Check each direct base class
			for (const auto& Base : CurrentClass->bases())
			{
				if (const auto* BaseClassDecl =
				        Base.getType()->getAsCXXRecordDecl())
				{
					// Check if FieldClass is an ancestor of this base class (or
					// is this base class itself)
					if (BaseClassDecl == FieldClass ||
					    isAncestorOf(FieldClass, BaseClassDecl))
					{
						// Add the offset of this base class
						uint64_t BaseOffsetBytes =
						    CurrentLayout.getBaseClassOffset(BaseClassDecl)
						        .getQuantity();
						TotalOffset += BaseOffsetBytes;
						CurrentClass = BaseClassDecl;
						Found = true;
						break;
					}
				}
			}

			if (!Found)
			{
				// This shouldn't happen if the field was collected correctly
				return 0;
			}
		}

		// Now add the field's offset within its declaring class
		const ASTRecordLayout& FieldLayout =
		    Context.getASTRecordLayout(FieldClass);
		uint64_t FieldOffsetBits =
		    FieldLayout.getFieldOffset(Field->getFieldIndex());
		TotalOffset += FieldOffsetBits / 8;

		return TotalOffset;
	}

	bool isAncestorOf(const CXXRecordDecl* Ancestor,
	                  const CXXRecordDecl* Descendant)
	{
		if (Ancestor == Descendant)
		{
			return true;
		}

		for (const auto& Base : Descendant->bases())
		{
			if (const auto* BaseClassDecl =
			        Base.getType()->getAsCXXRecordDecl())
			{
				if (isAncestorOf(Ancestor, BaseClassDecl))
				{
					return true;
				}
			}
		}

		return false;
	}

	void generateClassReflection(const CXXRecordDecl* ClassDecl,
	                             llvm::raw_ostream& OS)
	{
		std::string ClassName = ClassDecl->getNameAsString();

		// Collect fields including inherited ones
		std::vector<const FieldDecl*> Fields;
		collectFieldsRecursively(ClassDecl, Fields);

		// Constexpr array of field info
		OS << "template<> inline constexpr std::array<Field, " << Fields.size()
		   << "> fields<" << ClassName << "> = {{\n";

		for (const auto* Field : Fields)
		{
			std::string FieldName = Field->getNameAsString();

			// Calculate the field's offset in the derived class
			uint64_t FieldOffsetBytes =
			    calculateFieldOffsetInDerivedClass(ClassDecl, Field);

			uint64_t FieldSizeBits = Context.getTypeSize(Field->getType());
			uint64_t FieldSizeBytes = FieldSizeBits / 8;

			// Handle pointer types
			uint64_t FieldRefBytes = 0;
			bool isPointer = Field->getType()->isPointerType();

			if (isPointer)
			{
				// Get referenced type
				QualType pointeeType = Field->getType()->getPointeeType();

				// Assign size of type
				FieldRefBytes = Context.getTypeSize(pointeeType) / 8;
			}

			OS << "\t{\"" << FieldName << "\", ";
			OS << FieldOffsetBytes << ", ";
			OS << FieldSizeBytes << ", ";
			OS << FieldRefBytes << ", ";
			OS << isPointer << "},\n";
		}

		OS << "}};\n\n";
	}
};

class ReflectionConsumer : public ASTConsumer
{
  private:
	ReflectionVisitor Visitor;

  public:
	explicit ReflectionConsumer(ASTContext& Context) : Visitor(Context) {}

	void HandleTranslationUnit(ASTContext& Context) override
	{
		Visitor.TraverseDecl(Context.getTranslationUnitDecl());
		Visitor.GenerateReflectionData();
	}
};

class ReflectionAction : public PluginASTAction
{
  protected:
	std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& CI,
	                                               llvm::StringRef) override
	{
		return std::make_unique<ReflectionConsumer>(CI.getASTContext());
	}

	bool ParseArgs(const CompilerInstance&,
	               const std::vector<std::string>&) override
	{
		return true;
	}

	PluginASTAction::ActionType getActionType() override
	{
		return AddAfterMainAction;
	}
};

// Register the plugin
static FrontendPluginRegistry::Add<ReflectionAction>
    X("reflect", "Generate C++ reflection data");