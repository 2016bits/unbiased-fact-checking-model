import json

def analyze_domain(data_path, data_type):
    data_path = data_path.replace("[DATA]", data_type)
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    domain_dict = {}
    sup_domain_dict = {}
    ref_domain_dict = {}
    nei_domain_dict = {}
    for data in dataset:
        domain = data["domain"]
        label = data["label"]

        if domain in domain_dict:
            domain_dict[domain] += 1
        else:
            domain_dict[domain] = 1
            
        if label == 0:
            if domain in sup_domain_dict:
                sup_domain_dict[domain] += 1
            else:
                sup_domain_dict[domain] = 1
        elif label == 1:
            if domain in ref_domain_dict:
                ref_domain_dict[domain] += 1
            else:
                ref_domain_dict[domain] = 1
        elif label == 2:
            if domain in nei_domain_dict:
                nei_domain_dict[domain] += 1
            else:
                nei_domain_dict[domain] = 1

    print("****************{}****************".format(data_type))
    for key in domain_dict:
        print("{}:total num: {}".format(key, domain_dict[key]))
        if key in sup_domain_dict:
            print("domain: {}, supports num: {}".format(key, sup_domain_dict[key]))
        if key in ref_domain_dict:
            print("domain: {}, refutes num: {}".format(key, ref_domain_dict[key]))
        if key in nei_domain_dict:
            print("domain: {}, nei num: {}".format(key, nei_domain_dict[key]))


def main():
    data_path = "data/raw/[DATA].json"
    
    analyze_domain(data_path, "train")
    analyze_domain(data_path, "dev")
    analyze_domain(data_path, "test")

if __name__ == "__main__":
    main()