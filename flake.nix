{
  description = "Rust development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      nixpkgs,
      rust-overlay,
      flake-utils,
      ...
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
          buildInputs = [
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
        pytorchShell = defaultShell // {
          buildInputs = defaultShell.buildInputs ++ [ pkgs.python3Packages.torch-bin ];
          shellHook = ''
            export LIBTORCH_USE_PYTORCH=1
          ''
          + defaultShell.shellHook
          + ''
            echo "Python: $(python --version)"
            echo "Torch: $(python -c 'import torch; print(torch.__version__)')"
          '';
        };
      in
      {
        devShells = {
          default = pkgs.mkShell defaultShell;
          pytorch = pkgs.mkShell pytorchShell;
          pytorch-cuda = pkgs.mkShell (
            pytorchShell
            // {
              buildInputs =
                pytorchShell.buildInputs
                ++ (with pkgs.cudaPackages; [
                  cudatoolkit
                  nccl
                ]);
            }
          );
        };
      }
    );
}
