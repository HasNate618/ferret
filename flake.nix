{
  description = "ferret: local-first semantic file search";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAll = nixpkgs.lib.genAttrs systems;
    in {
      packages = forAll (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          ferret = pkgs.rustPlatform.buildRustPackage {
            pname = "ferret";
            version = "0.1.0";
            src = pkgs.lib.cleanSource ./.;
            cargoLock.lockFile = ./Cargo.lock;
            nativeBuildInputs = [ pkgs.pkg-config pkgs.makeWrapper ];
            # reqwest loads the native CA bundle at Client::build() time
            # (the test suite builds clients against httpmock). The build
            # sandbox ships no system bundle, so point at nixpkgs' one.
            SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
            # poppler-utils on PATH: pdfinfo/pdftotext/pdftoppm for PDFs.
            # No ImageMagick here (see nixosModules.ferret heicMagick).
            postInstall = ''
              wrapProgram $out/bin/ferret --prefix PATH : ${
                pkgs.lib.makeBinPath [ pkgs.poppler-utils ]
              }
            '';
            meta = {
              description = "Local-first semantic multimodal file search";
              mainProgram = "ferret";
            };
          };
        in {
          inherit ferret;
          default = ferret;
        });

      devShells = forAll (system:
        let pkgs = nixpkgs.legacyPackages.${system};
        in {
          default = pkgs.mkShell {
            packages = [
              pkgs.rustc
              pkgs.cargo
              pkgs.clippy
              pkgs.rustfmt
              pkgs.pkg-config
              pkgs.poppler-utils
            ];
          };
        });

      overlays.default = final: prev: {
        ferret = self.packages.${final.system}.ferret;
      };

      nixosModules.ferret = import ./nix/module.nix;
    };
}
