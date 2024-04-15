import json

settings_filename = "settings.json"
settings = dict(outpath="out")


if __name__=="__main__":
    with open(settings_filename, "w") as f:
        json.dump(settings, settings_filename, indent=2)
    print(f"{settings_filename} is generated.")