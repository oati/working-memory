{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      perSystem = { pkgs, ... }: {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            (python312.withPackages (p: with p; [
              jax
              equinox
              matplotlib

              # LSP tools
              python-lsp-server
              python-lsp-ruff
              python-lsp-black
              pylsp-mypy
            ]))
          ];
        };
      };
    };
}
