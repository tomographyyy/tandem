import json

settings_filename = "settings.json"
settings = dict(outpath="out")

def generate():
    with open(settings_filename, "w") as f:
        json.dump(settings, settings_filename, indent=2)
    print(f"{settings_filename} is generated.")    


if __name__=="__main__":
    generate()