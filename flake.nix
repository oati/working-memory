{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    pyedifice = {
      url = "github:pyedifice/pyedifice/v0.3.2";
      # url = "github:pyedifice/pyedifice/a4558f4f95a18f5b0ee7a9a20d2c661166e50139";
      flake = false;
    };
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      perSystem = { pkgs, ... }:
      let
        # Edifice package
        pyedifice = { buildPythonPackage, poetry-core, pyside6, qasync, ... }: buildPythonPackage {
          pname = "pyedifice";
          version = "v0.3.0";
          pyproject = true;
          src = inputs.pyedifice;
          buildInputs = [ poetry-core ];
          propagatedBuildInputs = [
            pyside6
            qasync
          ];
          doCheck = false;
        };
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            (python311.withPackages (p: with p; [
              jax
              equinox
              matplotlib
              pyqtgraph
              (p.callPackage pyedifice {})
              watchdog

              # LSP tools
              python-lsp-server
              python-lsp-ruff
              python-lsp-black
              pylsp-mypy
            ]))
          ];

          QT_PLUGIN_PATH = "${pkgs.qt6.qtwayland}/lib/qt-6/plugins";
          QT_QPA_PLATFORM = "wayland";
        };
      };
    };
}
