#pragma once

#include <array>
#include <cstddef>
#include <string>

#define REFLECT __attribute__((annotate("reflect")))

namespace reflection
{

struct Field
{
	std::string_view name;
	size_t offset;
	size_t size;
	size_t refSize;
	bool isPointer;

	template <typename T> void*& ref(T& instance) const
	{
		return *(void**)((std::byte*)(&instance) + offset);
	}
};

template <typename T> constexpr std::array<Field, 0> fields;

}