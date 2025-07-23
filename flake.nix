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
          (final: prev: {
            cudaPackages = prev.cudaPackages // {
              nccl = prev.cudaPackages.nccl.overrideAttrs (oldAttrs: rec {
                version = "2.27.6-1";
                src = prev.fetchFromGitHub {
                  owner = "NVIDIA";
                  repo = "nccl";
                  rev = "v${version}";
                  hash = "sha256-/BiLSZaBbVIqOfd8nQlgUJub0YR3SR4B93x2vZpkeiU=";
                };
                postPatch = ''
                  patchShebangs ./src/device/generate.py
                  patchShebangs ./src/device/symmetric/generate.py
                '';
              });
            };
          })
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

        python = pkgs.python312.override {
          packageOverrides = pythonSelf: pythonSuper: {
            torch-bin =
              let
                cudaVersion = "128";
                version = "2.8.0";
                srcs = {
                  "x86_64-linux-312" = pkgs.fetchurl {
                    url = "https://download.pytorch.org/whl/test/cu${cudaVersion}/torch-${version}%2Bcu${cudaVersion}-cp312-cp312-manylinux_2_28_x86_64.whl";
                    hash = "sha256-Q1T8Bbt5sgjWmVoEyhzu9qlUexxDNENVdDU9OBxVCHw=";
                  };
                };
                pyVerNoDot = builtins.replaceStrings [ "." ] [ "" ] pythonSelf.python.pythonVersion;
                unsupported = throw "Unsupported system";
              in
              pythonSuper.torch-bin.overrideAttrs (oldAttrs: rec {
                inherit version;
                src = srcs."${pkgs.stdenv.system}-${pyVerNoDot}" or unsupported;
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
