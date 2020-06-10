import json



def write_config(configs,file_name='config.json'):
    print(json.dumps(configs))
    with open(file_name, 'w') as outfile:
            json.dump(configs, outfile)


def read_config(file_name='config.json'):
    try:
        with open(file_name) as json_file:
            data = json.load(json_file)
            #print(data)
            return data
    except:
        print("Unable to open config file:",file_name)
        return {}
