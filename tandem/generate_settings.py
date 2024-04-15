import json

settings_filename = "settings.json"
settings = dict(outpath="outdir=output_0101-1234567-forward_C1S1B1GR030_Sheehan_Manning0.025_NL",
                job_id="1234567",
                job_name="job_name",
                )

def generate():
    with open(settings_filename, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"{settings_filename} is generated.")    


if __name__=="__main__":
    generate()