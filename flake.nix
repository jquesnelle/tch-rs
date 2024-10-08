{
  description = "tch-rs development flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs @ {
    flake-parts,
    rust-overlay,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      perSystem = {system, ...}: let
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [(import rust-overlay)];

          config.allowUnfree = true;
          config.cudaSupport = true;
          config.cudaVersion = "12.4";
        };
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = ["rust-src"];
        };

        torch = pkgs.libtorch-bin.dev.overrideAttrs (old: let
          version = "2.4.1";
          cuda = "124";
        in {
          version = version;
          src = pkgs.fetchzip {
            name = "libtorch-cxx11-abi-shared-with-deps-${version}-cu${cuda}.zip";
            url = "https://download.pytorch.org/libtorch/cu${cuda}/libtorch-cxx11-abi-shared-with-deps-${version}%2Bcu${cuda}.zip";
            hash = "sha256-/MKmr4RnF2FSGjheJc4221K38TWweWAtAbCVYzGSPZM=";
          };
        });
      in {
        devShells.default = pkgs.mkShell {
          env = {
            CUDA_ROOT = pkgs.cudaPackages.cudatoolkit.out;
            LIBTORCH = torch.out;
            LIBTORCH_INCLUDE = torch.dev;
            LIBTORCH_LIB = torch.out;
          };

          nativeBuildInputs = [
            pkgs.pkg-config
          ];

          buildInputs = [torch rustToolchain pkgs.openssl] ++ (with pkgs.cudaPackages; [cudatoolkit cuda_cudart nccl]);
        };
      };
    };
}
