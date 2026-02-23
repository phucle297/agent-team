{
  description = "Dev env";

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            nodejs
            python314
            python314Packages.pip
            python314Packages.virtualenv
            git
            gnumake
            bash
            coreutils
            jq
            gnused
            gnugrep
          ];
        };
      }
    );
}
