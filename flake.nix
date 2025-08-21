{
  description = "Rust development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [
          (import rust-overlay)
        ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
          config.cudaSupport = true;
          config.cudaVersion = "12.8";
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [
            "rust-src"
            "clippy"
          ];
        };

        libnvshmem = pkgs.stdenv.mkDerivation rec {
          pname = "libnvshmem";
          version = "3.3.20";

          src = pkgs.fetchurl {
            url = "https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-${version}_cuda12-archive.tar.xz";
            hash = "sha256-dXRstC611auvROEo2SOKFiO2TNTUm8LE2O+cYI1Gx+E=";
          };

          nativeBuildInputs = with pkgs; [ autoPatchelfHook ];
          buildInputs =
            with pkgs;
            [
              stdenv.cc.cc.lib
              libpciaccess
              libfabric
              ucx
              pmix
              mpi
            ]
            ++ (with cudaPackages; [
              cuda_cudart
              cuda_nvcc
            ]);

          installPhase = ''
            runHook preInstall

            mkdir -p $out/{lib,include,bin,share}

            cp -r lib/* $out/lib/

            cp -r include/* $out/include/

            cp -r bin/* $out/bin/

            cp -r share/* $out/share/

            cp LICENSE $out/share/

            runHook postInstall
          '';

          postFixup = ''
            # Fix RPATH for binaries
            find $out/bin -type f -executable | while read -r file; do
              if [[ -f "$file" && ! -L "$file" ]]; then
                patchelf --set-rpath "${pkgs.lib.makeLibraryPath buildInputs}:$out/lib" "$file" 2>/dev/null || true
              fi
            done

            # Fix RPATH for shared libraries
            find $out/lib -name "*.so*" | while read -r file; do
              if [[ -f "$file" && ! -L "$file" ]]; then
                patchelf --set-rpath "${pkgs.lib.makeLibraryPath buildInputs}:$out/lib" "$file" 2>/dev/null || true
              fi
            done
          '';

          meta = with pkgs.lib; {
            description = "NVIDIA SHMEM (NVSHMEM) is a parallel programming interface based on OpenSHMEM";
            homepage = "https://developer.nvidia.com/nvshmem";
            license = licenses.unfree;
            platforms = [ "x86_64-linux" ];
            maintainers = [ ];
          };
        };

        python = pkgs.python312.override {
          packageOverrides = pythonSelf: pythonSuper: {
            torch-bin =
              let
                pyCudaVer = builtins.replaceStrings [ "." ] [ "" ] pkgs.config.cudaVersion;
                version = "2.9.0.dev20250811";
                nightly = true;
                srcs = {
                  "x86_64-linux-312" = pkgs.fetchurl {
                    url = "https://download.pytorch.org/whl/${
                      if nightly then "nightly/" else ""
                    }cu${pyCudaVer}/torch-${version}%2Bcu${pyCudaVer}-cp312-cp312-manylinux_2_28_x86_64.whl";
                    hash = "sha256-N4y1ClwOFYz20p4SoLBuuB/zwqoAfeO4n8Ds9FZFpg0=";
                  };
                };
                pyVerNoDot = builtins.replaceStrings [ "." ] [ "" ] pythonSelf.python.pythonVersion;
                unsupported = throw "Unsupported system";
              in
              pythonSuper.torch-bin.overrideAttrs (oldAttrs: rec {
                inherit version;
                src = srcs."${pkgs.stdenv.system}-${pyVerNoDot}" or unsupported;

                buildInputs =
                  oldAttrs.buildInputs
                  ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
                    libnvshmem
                  ];
              });
          };
        };

        ocamlDeps = with pkgs.ocamlPackages; [
          ocaml
          base
          core
          stdio
          dune_3
          yaml

          ocaml_intrinsics_kernel
          sexplib0
          ctypes
          ctypes-foreign
          bigarray-compat
          integers
          stdlib-shims
        ];
        defaultShell = {
          buildInputs =
            with pkgs;
            [
              rustToolchain
            ]
            ++ ocamlDeps;

          shellHook = ''
            export OCAMLPATH="${pkgs.lib.makeSearchPath "lib/ocaml/${pkgs.ocamlPackages.ocaml.version}/site-lib" ocamlDeps}"
            echo "Rust + OCaml development environment"
            echo "Rust: $(rustc --version)"
            echo "Cargo: $(cargo --version)"
            echo "Dune: $(dune --version)"
            echo "OCaml: $(ocaml --version)"
          '';
        };
      in
      {
        devShells = {
          default = pkgs.mkShell defaultShell;
          pytorch = pkgs.mkShell (
            defaultShell
            // {
              buildInputs = defaultShell.buildInputs ++ [ python.pkgs.torch-bin ];
              shellHook =
                ''
                  export LIBTORCH_USE_PYTORCH=1
                ''
                + defaultShell.shellHook
                + ''
                  echo "Python: $(python --version)"
                  echo "Torch: $(python -c 'import torch; print(torch.__version__)')"
                '';
            }
          );
        };
      }
    );
}
