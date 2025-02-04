{
  description = "PUFFINN - Parameterless and Universal Fast FInding of Nearest Neighbors";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = {
    self,
    nixpkgs,
  }: let
    supportedSystems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];
    forEachSupportedSystem = f:
      nixpkgs.lib.genAttrs supportedSystems (system:
        f {
          pkgs = import nixpkgs {inherit system;};
        });
  in {
    devShells = forEachSupportedSystem ({pkgs}: {
      default = pkgs.mkShell {
        venvDir = ".venv";
        packages = with pkgs; [
          python310
          sqlite-interactive
          cmake
        ] ++ ( with python310Packages; [
          setuptools
          wheel
          venvShellHook
          numpy
          h5py
        ]);
      };
    });
  };
}
