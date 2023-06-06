load("@local_config_cuda//cuda:build_defs.bzl", "cuda_header_library")

cc_library(
    name = "hkv",
    hdrs = glob([
        "merlin_hashtable.cuh",
    ]),
    # strip_include_prefix = "include",
    includes = [
        "include/"
    ],
    visibility = ["//visibility:public"],
)