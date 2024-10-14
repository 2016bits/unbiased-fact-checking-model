import json
import openai
import argparse

from tqdm import tqdm

prompt = """The claims are rewritten based on the provided evidence to ensure that the modified claims are supported by the evidence.
original claim: On 15 October 2021, Chinese and foreign banking institutions announced their opposition to biodiversity conservation.
evidence: On 15 October, 36 banking institutions, 24 foreign banks and international organizations jointly published the Joint Declaration of Banking Financial Institutions in Support of Biodiversity Conservation to further strengthen support for biodiversity conservation. On that day, the Forum on the theme “Banking Financial Institutions in Support of Biodiversity Conservation” of the 2020 United Nations Conference on Biodiversity (first phase) was held in Kunming. On 15 October, at the fifteenth meeting of the Conference of the Parties to the Convention on Biological Diversity (COP15), the World Wide Fund for Nature (WWF), one of the co-sponsors, promoted and provided technical support to 36 banking institutions, 24 foreign banks and international organizations in their joint advocacy in support of biodiversity conservation.
new claim: On 15 October 2021, Chinese and foreign banking institutions declared their support for biodiversity conservation.

original claim: On 23 August, the National Health Board and the Ministry of Education issued the latest version of the programme for the prevention and control of new coronary pneumonia in higher schools, primary and secondary schools and childcare institutions, which requires teachers and students in higher education to give 48-hour nucleic acid certificates before returning to school.
evidence: On 23 August, according to the website of the National Health Commission, the National Health Commission and the Ministry of Education jointly issued a circular on the publication of a technical programme for the prevention and control of new coronary pneumonia in schools, small and medium schools and childcare institutions (version IV). The circular requires that schools have information on the health status and the journey of their staff 14 days before they return to school, and that teachers and students in higher education provide negative proof of nucleic acid testing within 48 hours before they return to school, and that they can be tested for nucleic acid on a separate basis, in accordance with local control requirements.
new claim: On 23 August, the National Health Board and the Ministry of Education issued the latest version of the programme for the prevention and control of new coronary pneumonia in higher schools, primary and secondary schools and childcare institutions, which requires teachers and students in higher education to provide 48-hour nucleic acid certificates before returning to school.

original claim: From 2020, card users will benefit from a basic vehicle toll policy of no less than 5 per cent.
evidence: The Ministry of Transport indicated that the basic policy of granting a vehicle toll discount of no less than 5 per cent to ETC vehicles is intended to encourage the use of ETC for vehicle installation, improve efficiency of access, promote energy efficiency and reduce highway operating costs. ETC card users are a means of providing non-cash payments to trucks on a cost-based basis, which can be used only for manual charges. Considering that the truck toll system is currently being adapted in various locations and that conditions are not yet in place to fully realize the non-remobilization fast charges for trucks, the Ministry of Transport stated that they have deployed and will continue to support their non-cash payment function by no less than 5 per cent of the basic toll policy for single card users until the end of 2019; however, as of 1 January 2020, they will no longer be eligible for the ETC preferential policy.
new claim: From 2020, card users will no longer benefit from a basic vehicle toll policy of no less than 5 per cent in principle.

original claim: [CLAIM]
evidence: [EVIDENCE]
new claim: 
"""

def define_gpt():
    # chatgpt api
    openai.api_type = 'azure'
    openai.api_base = 'https://1027gpt.openai.azure.com/'
    openai.api_version = '2023-03-15-preview'
    openai.api_key = '4cd2d1e5aa8849cca6050deaa5c0a277'

def llm(prompt, stop=["\n"]):
    response = openai.ChatCompletion.create(
            engine="ChatGPT",
            temperature=0,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop,
            messages=[{"role":"user","content":prompt}]
        )
    return response.choices[0].message.content

def main(args):
    define_gpt()
    f_tmp = open(args.tmp_path, 'w', encoding='utf-8')
    for datatype in ['dev', 'test']:
        print(datatype, file=f_tmp)
        in_path = args.in_path + datatype + ".json"
        with open(in_path, 'r', encoding='utf8') as f:
            dataset = json.load(f)
        refuted_dataset = [d for d in dataset if d['label']==1]
        
        results = []
        for data in tqdm(refuted_dataset):
            claim = data['translated_claim']
            evidence = data['translated_evidence']
            prompt_text = prompt.replace("[CLAIM]", claim).replace("[EVIDENCE]", evidence)
            try:
                result = llm(prompt_text)
            except:
                print(data['id'], " ", "error: ", data['translated_claim'])
                continue
            print(data['id'], ": ", result, file=f_tmp)
            results.append({
                "id": data['id'],
                "claim": data['claim'],
                "evidence": data['evidence'],
                "label": data['label'],
                "translated_claim": data['translated_claim'],
                "translated_evidence": data['translated_evidence'],
                "claim_refuted": result
            })
        
        out_path = args.out_path + datatype + ".json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    f_tmp.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="data/symmetric-CHEF/translated_")
    parser.add_argument("--demonstration_path", type=str, default="data/symmetric-CHEF/supported_negative_")
    parser.add_argument("--out_path", type=str, default="data/symmetric-CHEF/refuted_negative_")

    parser.add_argument("--tmp_path", type=str, default="data/symmetric-CHEF/tmp.txt")
    args = parser.parse_args()
    main(args)
