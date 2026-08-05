target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_values("cuda.rdc", false)
    add_rules("cuda")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC")
    end

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_values("cuda.rdc", false)
    add_rules("cuda")
    set_warnings("all", "error")
    add_deps("llaisys-tensor")
    add_deps("llaisys-device-nvidia")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC")
    end

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
