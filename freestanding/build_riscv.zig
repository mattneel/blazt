const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .cpu_arch = .riscv32,
            .os_tag = .freestanding,
        },
    });

    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseFast,
    });

    // Freestanding modules (replacements for build-time generated ones)
    const cpu_cache = b.createModule(.{
        .root_source_file = b.path("cpu_cache.zig"),
        .target = target,
    });

    const build_options = b.createModule(.{
        .root_source_file = b.path("build_options.zig"),
        .target = target,
    });

    // The main blazt module (same structure as main build.zig)
    // Uses root.zig which re-exports all internal modules via relative imports
    const blazt = b.createModule(.{
        .root_source_file = b.path("../src/root.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "cpu_cache", .module = cpu_cache },
            .{ .name = "build_options", .module = build_options },
        },
    });

    // Create library using newer API
    const ffi_module = b.createModule(.{
        .root_source_file = b.path("ffi.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "blazt", .module = blazt },
        },
    });

    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "blazt_riscv",
        .root_module = ffi_module,
    });

    b.installArtifact(lib);
}
