{
  description = "AI Agent Team Dev Env";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python312                     # <--- Changed from 311 to 312
            python312Packages.pip
            python312Packages.virtualenv
            nodejs_22
            git
            jq
          ];

          shellHook = ''
            # If a 3.11 venv exists, remove it to avoid conflicts
            if [ -d ".venv" ] && [[ $(python --version) != *"3.12"* ]]; then
              echo "Cleaning up old Python 3.11 virtual environment..."
              rm -rf .venv
            fi

            if [ ! -d ".venv" ]; then
              echo "Creating new Python 3.12 virtual environment..."
              python -m venv .venv
            fi
            
            source .venv/bin/activate
            
            echo "Installing dependencies..."
            pip install -q langgraph langchain-google-genai langchain-anthropic python-dotenv

            echo "✅ Agent Environment Ready (Python 3.12)"
          '';
        };
      }
    );
}
