from std.sys import CompilationTarget


@always_inline
def has_avx512_bf16() -> Bool:
    return CompilationTarget._has_feature["avx512bf16"]()


@always_inline
def has_amx_int8() -> Bool:
    return CompilationTarget._has_feature["amx-int8"]()
