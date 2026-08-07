-- MetaX GPU backend build targets
-- All MetaX code uses native API (.mc files compiled with mxcc -x maca)

local MACA_PATH = os.getenv("MACA_PATH") or "/opt/maca"
local MXCC = path.join(MACA_PATH, "mxgpu_llvm/bin/mxcc")

rule("metax.macac")
    set_extensions(".mc")
    on_build_file(function (target, sourcefile, opt)
        import("core.project.depend")
        local objectfile = target:objectfile(sourcefile)
        local dependfile = target:dependfile(objectfile)
        os.mkdir(path.directory(objectfile))
        local argv = {
            "-x", "maca",
            "-offload-arch", "native",
            "--maca-path=" .. MACA_PATH,
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-I" .. path.join(MACA_PATH, "include"),
            "-I" .. path.join(os.projectdir(), "include"),
            "-I" .. path.join(os.projectdir(), "src"),
            "-c", sourcefile,
            "-o", objectfile,
        }
        depend.on_changed(function ()
            vprint("compiling.$(mode) %s", sourcefile)
            os.vrunv(MXCC, argv)
        end, {dependfile = dependfile, files = {sourcefile}, changed = target:is_rebuilt()})
        table.insert(target:objectfiles(), objectfile)
    end)
