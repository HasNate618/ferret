{ config, lib, pkgs, ... }:

let
  cfg = config.services.ferret;
  toml = pkgs.formats.toml { };
in {
  options.services.ferret = {
    enable = lib.mkEnableOption "ferret local-first semantic file search";

    package = lib.mkPackageOption pkgs "ferret" { };

    settings = lib.mkOption {
      type = toml.type;
      default = { };
      example = {
        index.roots = [ "/home/alice/notes" ];
        embedder.backend = "llama_server";
      };
      description = ''
        ferret configuration, written verbatim to
        {file}`/etc/ferret/config.toml`. Keys mirror the TOML sections
        (`index`, `embedder`, `search`, `chunk`, `extract`, `files`).
        An explicit `--config` flag or a user config file still
        overrides these values at runtime.
      '';
    };

    heicMagick = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Adds ImageMagick (`magick`/`convert`) to the system PATH so
        `files.heic = "magick"` can convert HEIC/HEIF. Off by default:
        the default package only wraps poppler-utils.
      '';
    };
  };

  # No daemon: ferret is a CLI. This module only installs the package
  # and the system config file.
  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ cfg.package ]
      ++ lib.optional cfg.heicMagick pkgs.imagemagick;

    environment.etc."ferret/config.toml".source =
      toml.generate "ferret-config.toml" cfg.settings;
  };
}
