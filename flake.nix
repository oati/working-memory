{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = inputs: let
    inherit (inputs.nixpkgs) lib;

    eachSystem = lib.genAttrs lib.systems.flakeExposed;
    pkgsFor = eachSystem (
      system:
        import inputs.nixpkgs {localSystem.system = system;}
    );
  in {
    devShells =
      lib.mapAttrs (system: pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            (python314.withPackages (p:
              with p; [
                jax
                equinox
                matplotlib
                tqdm

                # LSP tools
                python-lsp-server
                python-lsp-ruff
                python-lsp-black
                pylsp-mypy

                # jupyterlab
                jupyterlab
                jupyterlab-lsp
                ipywidgets
                # jupyterlab-widgets
                # jupyterlab-execute-time
              ]))
            ffmpeg
          ];
        };
      })
      pkgsFor;
  };
}
