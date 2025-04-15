import pm4py
import os
import time
import json
import traceback

prompt = "I need to train a LLM to recognize timestamp formats. This can be done if you generate a list of timestamps and the associated format that should be used for the parsing. Please produce a JSON containing two keys: 'values' (which is the list of timestamp values) and 'format' (containing the recognized format). Please use your fantasy to generate an interesting timestamp format, but make sure that is a standard, recognized, format (not something imaginary). Please put the JSON between the tags ```json and ```"

i = 0
while i < 200:
    print(i)
    output_file_path = os.path.join("data", str(i).zfill(5)+".txt")

    if not os.path.exists(output_file_path):
        try:
            resp = pm4py.llm.openai_query(prompt, api_key=open("../../openai_api.txt", "r").read().strip(), openai_model="gpt-4.1-mini")
            resp = resp.split("```json")[-1].split("```")[0]
            print(resp)
            resp = json.loads(resp)
            if not ("values" in resp and "format" in resp):
                raise Exception("incomplete: "+str(resp))
            output_file = open(output_file_path, "w")
            json.dump(resp, output_file, indent=2)
            output_file.close()
            i = i + 1
        except:
            traceback.print_exc()
            time.sleep(10)
    else:
        i = i + 1
