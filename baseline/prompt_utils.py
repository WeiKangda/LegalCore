import re
from post_processing.utils import load_jsonl, append_to_jsonl, process_coreference, create_coreference_clusters, replace_elements_with_mentions, mentions_to_clusters
import os
from tqdm import tqdm
def extract_event_triggers_with_spans_fixed(text):
    """
    Extract event triggers, their indices, and spans from the input text.
    Properly handle punctuation, single quotes, and numeric formats (e.g., "1.4").

    Args:
        text (str): Input text containing event triggers in the format {EXX trigger_word}.

    Returns:
        list of dict: Each dict contains 'Span' and 'Trigger' for each event.
    """
    # Replace {EXX trigger_word} with trigger_word
    cleaned_text = re.sub(r"\{E\d+\s+([^\}]+)\}", r"\1", text)

    # Use regex to split words while treating numeric formats as single tokens
    words = re.findall(r'\b\w+(?:\.\w+)?\b', cleaned_text)

    matches = list(re.finditer(r"\{E\d+\s+([^\}]+)\}", text))

    # Extract events and their spans
    events = []
    current_index = 0
    print(words)

    for word in words:

        for match in matches:
            trigger_word = match.group(1)
            if trigger_word==word:
                # Calculate span range
                start_index = max(0, current_index - 2)
                end_index = min(len(words) - 1, current_index + 2)
                span = f"{start_index}-{end_index}"
                events.append({
                    "Span": span,
                    "Trigger": trigger_word
                })
                break
        print(current_index)

        current_index += 1

    return events

def coreference_fewshot_prompt_generate(prompt,inference_mode):
    data_path = "../data/data.jsonl"

    all_data = load_jsonl(data_path)
    all_gold = []
    all_data = all_data[:3]
    all_text = []
    for data in tqdm(all_data):
        # clusters=mentions_to_clusters(data["events"])
        clusters = []
        mentions = data["events"]
        text = data["singleton_text"]
        all_text.append(text)
        for mention_group in mentions:
            cluster = []
            if len(mention_group) > 1:
                for mention in mention_group["mention"]:
                    cluster.append((mention["singleton_id"], mention["trigger_word"]))
                clusters.append(cluster)
        all_gold.append(clusters)
    def coreference_linking(clusters):
        prompt=""
        for cluster in clusters:
            first_mention_id=cluster[0][0]
            for mention in cluster[1:]:
                prompt+=f"""{first_mention_id} COREFERENCE {mention[0]}\n"""
        return prompt

    one_shot_prompt=f"""Please analyze the following text to detect all coreference relations among events. \
Two events have a coreference relation if they refer to the same event in space and time. \
Coreference relation is symmetrical, i.e., non-directional: If A coreference B, then B coreference A. \
It is also transitive: If A coreference B and B coreference C, then A coreference C. \
An event should only be linked with one of its nearest antecedents occurring before itself. \
There is no need to link with multiple antecedents as this information is redundant. \
All events are denoted as {{EXX trigger_word}} in the text. Format your response as: 'EXX COREFERENCE EXX'. \
Hint: Coreferential event mentions usually have the same trigger_word.\
If multiple coreference relations exist, list each relation on a new line. \
If no coreference relation is detected, simply return 'None'. Please respond concisely and directly to the point, avoiding unnecessary elaboration or verbosity.
                            
**Example:**
Text:{all_text[0]}
Response: {coreference_linking(all_gold[0])}

Now analyze the following text:
Text: {str(prompt)} 
Response:"""
    two_shot_prompt=f"""Please analyze the following text to detect all coreference relations among events. \
Two events have a coreference relation if they refer to the same event in space and time. \
Coreference relation is symmetrical, i.e., non-directional: If A coreference B, then B coreference A. \
It is also transitive: If A coreference B and B coreference C, then A coreference C. \
An event should only be linked with one of its nearest antecedents occurring before itself. \
There is no need to link with multiple antecedents as this information is redundant. \
All events are denoted as {{EXX trigger_word}} in the text. Format your response as: 'EXX COREFERENCE EXX'. \
Hint: Coreferential event mentions usually have the same trigger_word.\
If multiple coreference relations exist, list each relation on a new line. \
If no coreference relation is detected, simply return 'None'. Please respond concisely and directly to the point, avoiding unnecessary elaboration or verbosity.
                            
**Example 1:**
Text:{all_text[1]}
Response: {coreference_linking(all_gold[1])}

**Example 2:**
Text:{all_text[1]}
Response: {coreference_linking(all_gold[1])}

Now analyze the following text:
Text: {str(prompt)} 
Response:"""
    print(two_shot_prompt)
    if inference_mode=="one_shot":
        return one_shot_prompt
    elif inference_mode=="two_shot":
        return two_shot_prompt

if __name__ == "__main__":
    # print(os.getcwd())  # 输出当前工作目录
    #
    # # data_path="/Users/kaychan/PycharmProjects/Legal-Coreference/annotation_validation/jonathan_annotations/data.jsonl"
    # data_path="../annotation_validation/jonathan_annotations/data.jsonl"
    #
    # all_data = load_jsonl(data_path)
    # all_gold=[]
    # all_data=all_data[:1]
    # all_text=[]
    # for data in tqdm(all_data):
    #     # clusters=mentions_to_clusters(data["events"])
    #     # clusters = []
    #     # mentions=data["events"]
    #     # text=data["singleton_text"]
    #     # all_text.append(text)
    #     # for mention_group in mentions:
    #     #     cluster = []
    #     #     if len(mention_group)>1:
    #     #         for mention in mention_group["mention"]:
    #     #             cluster.append((mention["singleton_id"], mention["trigger_word"]))
    #     #         clusters.append(cluster)
    #     # all_gold.append(clusters)
    #     # print("Gold mentions:" + str(clusters))
    #     # print("########################")
    #     # print()
    #     coreference_fewshot_prompt_generate("test","one_shot")
    #     cnt=0
    #     # for cluster in clusters:
    #     #     if len(cluster)>1:
    #     #         cnt+=1
        #         print(cluster)
    prompt="test"
    print(r"""Please analyze the following text to detect all coreference relations among events. Two events have a coreference relation if they refer to the same event in space and time. Coreference relation is symmetrical, i.e., non-directional: If A coreference B, then B coreference A. It is also transitive: If A coreference B and B coreference C, then A coreference C. An event should only be linked with one of its nearest antecedents occurring before itself. There is no need to link with multiple antecedents as this information is redundant. All events are denoted as {EXX trigger_word} in the text. Format your response as: 'EXX COREFERENCE EXX'. Hint: Coreferential event mentions usually have the same trigger_word.If multiple coreference relations exist, list each relation on a new line. If no coreference relation is detected, simply return 'None'. Please respond concisely and directly to the point, avoiding unnecessary elaboration or verbosity.
                            
**Example 1:**
Text:VIRTUAL ITEM PROCESSING SYSTEMS, INC. 2525 Northwest Expressway, #105 Oklahoma City, Oklahoma 73112 OUTSOURCING AGREEMENT BETWEEN VIRTUAL ITEM PROCESSING SYSTEMS, INC. And BROKERS NATIONAL LIFE ASSURANCE COMPANY E - 4 OUTSOURCING AGREEMENT This Outsourcing Agreement (" Agreement") is {E0 executed} as of this 1 st day of May 2006 , by and between Virtual Item Processing Systems, Inc. ("VIP"), with its principal place of office at 2525 NW Expressway, Suite 105 Oklahoma City, Oklahoma 73112, and Brokers National Life Assurance Company ("BNL"), with its principal place of office at 7010 Hwy. 71 W., Suite 100, Austin, Texas 78735. WHEREAS, VIP is {E1 engaged} in the business of {E2 providing} Electronic Data Processing {E3 services} ("EDP {E4 Services} ") and related {E5 consultation} and {E6 services} to insurance companies pursuant to computer software systems {E7 developed} and {E8 owned} by VIP , (the "VIP System"); WHEREAS, BNL is an insurance company {E9 domiciled} in the State of Arkansas and {E10 licensed} to {E11 do} business in numerous additional states; and WHEREAS, VIP desires to {E12 provide} EDP Services to BNL; and WHEREAS, BNL desires to {E13 obtain} EDP {E14 services} from VIP for the {E15 processing} and {E16 administration} of its insurance policies; NOW, THEREFORE, in consideration of the above premises and in consideration of other good and valuable consideration, the {E17 receipt} and sufficiency is hereby {E18 acknowledged} , the parties {E19 agree} as follows : 1. PURCHASE OF EQUIPMENT. BNL at its expense shall {E20 obtain} , {E21 install} , {E22 maintain} and {E23 upgrade} as necessary any and all hardware, software, data and telephone lines, other communications equipment and any other equipment (hereinafter collectively referred to as the "Equipment") which it {E24 determines} is necessary to {E25 allow} it to {E26 use} and {E27 access} the VIP System pursuant to the terms of this Agreement. Such Equipment shall be fully compatible with the VIP System. VIP will {E28 provide} BNL such information as is reasonably necessary to {E29 allow} BNL to {E30 acquire} all such Equipment which {E31 meets} the requirements of this paragraph. If {E32 requested} by BNL and at BNL's expense, VIP shall {E33 inspect} all such Equipment and {E34 acknowledge} its compatibility in writing prior to its use with the VIP System. 2. VIP's EQUIPMENT AND SERVICES. A. During the term of this Agreement, VIP shall {E35 provide} BNL such access as necessary to the VIP System to {E36 allow} BNL to {E37 attach} one data communication line and up to seventy (70) addressable data communications devices to {E38 said} VIP System. Should BNL {E39 desire} to {E40 attach} additional communication lines or additional communication devices to the VIP System, BNL shall {E41 pay} to {E42 VIP} the additional fees set forth in paragraph 5(F) of this Agreement. B. VIP, at its sole discretion and expense, may, but is not {E43 obligated} to, {E44 make} appropriate {E45 enhancements} to the VIP System. Any such {E46 enhancements} shall be {E47 deemed} to be {E48 included} in the EDP {E49 Services} and VIP System to be {E50 provided} to BNL, whether {E51 developed} by VIP before or during the time when {E52 services} are to be {E53 provided} by VIP pursuant to this Agreement. During the term of this Agreement, VIP shall be responsible at its expense for the proper {E54 maintenance} and {E55 documentation} of the VIP System. 3 .SCHEDULED AND {E56 UNSCHEDULED} DOWN TIME. BNL {E57 acknowledges} that there will be {E58 scheduled} downtime for the routine preventive {E59 maintenance} of VIP's System {E60 performed} by either VIP or its vendors. VIP shall {E61 give} BNL reasonable advance notice of all such scheduled downtime. BNL further {E62 acknowledges} that there will also be {E63 unscheduled} down-time that might {E64 occur} as a result of electrical power {E65 failures} and equipment {E66 failures} and other {E67 acts} outside of the control of VIP as {E68 contemplated} in paragraph 16(J). In the event that any such down-time {E69 extends} for more than two (2) consecutive working days , VIP, at its expense, will {E70 make} available to BNL access to a backup facility {E71 designated} by VIP for the {E72 continued} {E73 processing} of BNL's business. To {E74 ensure} that a backup facility will be available in E - 5 case of such a {E75 failure} , VIP will {E76 maintain} disaster and/or business interruption insurance adequate to {E77 establish} alternate site {E78 processing} , as {E79 provided} for in paragraph 12(A) of this Agreement. 4. INCLUDED SERVICES IN THE VIP SYSTEM. It is {E80 agreed} and {E81 understood} by BNL that: A. It has {E82 reviewed} and {E83 inspected} the VIP System {E84 existing} as of the Effective Date of this Agreement, which VIP System {E85 includes} (i) a New Business System, (ii) a Policy Administration System, (iii) an Agency Administration System, (iv) a Financial Administration System. (v) a Claims System, (vi) a Vendor Provider System, (vii) a Transaction Tracking System and (viii) a Mail Tracking System; B. Such VIP System as {E86 identified} in paragraph 4(A) is adequate to {E87 meet} the needs of BNL; C. VIP shall {E88 provide} EDP {E89 Services} to BNL for such Initial Policies and policies identical thereto and {E90 renewals} thereof by the {E91 use} of such VIP System {E92 existing} as of the Effective Date of this Agreement, except as such VIP System may be {E93 modified} from time to time by VIP , at the discretion of VIP; D. BNL has {E94 reviewed} the security system (Security System") {E95 included} in the VIP System {E96 existing} as of the Effective Date of this Agreement; E. BNL {E97 acknowledges} and {E98 agrees} that such Security System is adequate to {E99 protect} the confidential information and data of BNL {E100 processed} by the VIP System; F. BNL, throughout the term of this Agreement, shall be solely responsible for {E101 choosing} , {E102 implementing} and {E103 utilizing} any or all of such of the security measures and protections {E104 offered} by {E105 said} Security System for the {E106 use} of or access to the VIP System by any of its officers, directors, shareholders, employees and agents; G. VIP shall not have any duty to either {E107 monitor} or {E108 enforce} such security {E109 measures} and {E110 protections} {E111 chosen} , {E112 implemented} or {E113 utilized} by BNL; H. E - 6 BNL shall be solely responsible for any {E114 acts} or {E115 omissions} of any of its officers, directors, shareholders, employees and agents; I. Notwithstanding anything to the contrary in this Agreement, VIP at any time during the term of this Agreement may {E116 change} the platform upon which the VIP System is {E117 operated} and through which the EDP Services are {E118 provided} to BNL by VIP under this Agreement. Before VIP shall {E119 make} such platform {E120 change} VIP shall {E121 give} BNL prior reasonable written notice of such {E122 change} , and VIP's warranties under this Agreement shall {E123 continue} notwithstanding such {E124 change} and VIP {E125 agrees} to {E126 pay} any cost {E127 created} for or {E128 imposed} on BNL for equipment, {E129 training} or similar matters {E130 arising} from such {E131 change} . 5. PAYMENTS TO VIP. A. For EDP {E132 Services} {E133 provided} pursuant to this Agreement, BNL will {E134 pay} to {E135 VIP} the {E136 charges} set forth in the Payment Schedule {E137 attached} hereto as Schedule B: {E138 provided} however and notwithstanding anything to the contrary herein. The minimum monthly fee shall not be less than five thousand dollars ($5,000) per month (as applicable, "Minimum Fee"). B. For any additional VIP {E139 Services} {E140 provided} hereunder, BNL will {E141 pay} to {E142 VIP} the {E143 charges} set {E144 charges} set forth in the Payment Schedule {E145 attached} hereto as Schedule A. C. The fees due hereunder are subject to the {E146 following} provisions: 1. The fee for each new policy {E147 submitted} into the VIP System is set forth in Schedule B. 2. VIP will {E148 process} all policies that have thirteen (13) or more months {E149 expired} from their original policy date at the annual rates set forth in Schedule B. with a separate fee for each renewal base policy and each rider, for each plan, {E150 prorated} to the actual number of months each policy is {E151 represented} to be in force on the VIP System. Such representation of "policy status" {E152 includes} the "grace period" and "Late payment {E153 offer} " that each policy may {E154 enjoy} and in which case {E155 exceeds} a time frame not {E156 bound} by each policy's actual {E157 paid} for period. The payment amount for each group of policies in a rate category will be {E158 calculated} by {E159 determining} the actual number of policies and riders in force that are {E160 included} in the rate category, as set forth in Schedule B at the end of each calendar month and then {E161 multiplying} the number of policies by the base policy renewal amount and the number of riders by the rider renewal amount then {E162 adding} the totals together and {E163 dividing} the {E164 resulting} amount by twelve (12). The amounts {E165 calculated} for all rate categories are {E166 added} together and this amount is the fee payable in advance at the beginning of the month . 3. VIP shall not be {E167 obligated} to {E168 process} any {E169 amended} policies or new products that E - 7 are {E170 written} or {E171 acquired} by BNL unless and until the parties hereto have mutually {E172 executed} a {E173 written} addendum to this Agreement {E174 modifying} Schedule B to {E175 include} the fees for any such products. D. Any sum due VIP hereunder for which a time for {E176 payment} is not otherwise specified will be due and payable within ten (10) days after the date of the postmark for an invoice therefor from VIP. If BNL {E177 fails} to {E178 pay} any amount due within ten (10) days from the date of the postmark for the invoice, late {E179 charges} of 1-1/2% per month , or the maximum amount allowable by law, whichever is less, shall also {E180 become} payable by BNL to VIP. E. In addition to the communication line and devices which BNL is {E181 authorized} to {E182 attach} to the VIP System pursuant to paragraph 2 of this Agreement, BNL may, for the monthly fee(s) hereinafter set forth, {E183 attach} additional communication lines or the {E184 following} devices to the VIP System. The monthly fee(s) for such additional lines) or devices is as follows : 1. each communication line and {E185 adapter} $200, 2. each visual station whether CRT, PC or similar device $25, 3. each addressable printer under 299 lines per minute ("LPM") $25 4. each addressable printer over 299 LPM $150. F. There are certain other expenses which are directly {E186 related} to VIP's {E187 performance} of this Agreement that are directly billable by VIP and payable by BNL. The purpose and intent of this provision is not to {E188 describe} all {E189 contemplated} {E190 charges} {E191 covered} by this provision, but rather to {E192 identify} some of the {E193 charges} that may {E194 fall} into this category . Such {E195 charges} {E196 include} but are not limited to the following : 1. Cost of all business forms, continuous or non-continuous {E197 used} by BNL; 2. All telephone {E198 calls} {E199 initiated} on behalf of BNL business and operations; 3. All travel, food and lodging expenses {E200 incurred} by VIP personnel {E201 related} to the {E202 performance} of this Agreement, subject to BNL's prior {E203 written} approval; 4. All postage and shipping expenses for materials {E204 used} by BNL; 5. All expenses {E205 incurred} for computer output micro-film "COM" which is {E206 contracted} by VIP with a service bureau independent of VIP , subject to BNL' s prior {E207 written} approval; 6. Any other {E208 charges} directly {E209 related} to BNL ' {E210 use} or benefit of the VIP System pursuant to this Agreement is subject to BNL ' prior {E211 written} approval. G. All sums due under this Agreement are payable in U.S. dollars. 6. PROPRIETARY AND RELATED RIGHTS. A. CLIENT DATA. Any original documents or files {E212 provided} to VIP hereunder by BNL ("BNL Data") are and shall {E213 remain} BNL's property and, upon the {E214 termination} of this Agreement for any reason, such BNL Data will be {E215 returned} to BNL by VIP, subject to E - 8 the terms hereof. Subject to paragraphs 4(F) and (G), VIP {E216 agrees} to {E217 make} the same effort to {E218 safeguard} such BNL Data as it {E219 does} in {E220 protecting} its own proprietary information. BNL Data will not be {E221 utilized} by VIP for any purpose other than those purposes {E222 related} to {E223 rendering} EDP {E224 Services} to BNL under this Agreement, nor will BNL Data or any part thereof be {E225 disclosed} to third parties by VIP , its employees or agents except for purposes {E226 related} to VIP's {E227 rendering} ofEDP {E228 Services} to BNL under this Agreement or as {E229 required} by law, regulation, or {E230 order} of a court or regulatory agency or other authority {E231 having} jurisdiction thereover. Notwithstanding the foregoing, VIP shall have the right to {E232 retain} in its possession all work papers and files {E233 prepared} by it in {E234 performance} of EDP {E235 Services} hereunder which may {E236 include} necessary copies of BNL Data. VIP shall have access to BNL Data, at reasonable times, during the term of this Agreement and thereafter for purposes {E237 related} to VIP's ' {E238 rendering} of EDP {E239 Services} to BNL pursuant to this Agreement, or as {E240 required} by law, regulation or {E241 order} of a court or regulatory agency or other authority {E242 having} jurisdiction thereover. Notwithstanding the foregoing, the confidentiality obligations set forth in this paragraph will not {E243 apply} to any information which (i) is or {E244 becomes} publicly available without {E245 breach} of this Agreement, (ii) is independently {E246 developed} by VIP outside the scope of this Agreement and without reference to the confidential information {E247 received} under this Agreement, or (iii) is rightfully {E248 obtained} by VIP from third parties which are not {E249 obligated} to {E250 protect} its confidentiality. 7. TERMINATION FOR CAUSE. This Agreement may be {E251 terminated} by the non- breaching party upon any of the {E252 following} {E253 events} : A. In the event that BNL {E254 fails} to {E255 pay} any sums of money due to VIP hereunder and does not {E256 cure} such {E257 default} within thirty (30) days after {E258 receipt} of written notice of such {E259 nonpayment} from VIP , {E260 provided} that if BNL {E261 notifies} VIP in writing that BNL {E262 disputes} a billing and BNL {E263 pays} any undisputed portion of such billing VIP shall not {E264 institute} formal {E265 proceedings} by {E266 arbitration} or judicial {E267 review} or {E268 terminate} this Agreement with respect to such {E269 disputed} billing until after VIP has {E270 afforded} BNL an opportunity for a {E271 meeting} to {E272 discuss} such {E273 dispute} . B. In the event that a party hereto {E274 breaches} any of the material terms, covenants or conditions of this Agreement (other than a {E275 breach} under paragraph (A) above) and {E276 fails} to {E277 cure} the same within thirty (30) days after {E278 receipt} of written notice of such {E279 breach} from the non-breaching party. C. In the event that a party hereto {E280 becomes} or is {E281 declared} insolvent or bankrupt, is the subject of any {E282 proceedings} {E283 relating} to its {E284 liquidation} , {E285 insolvency} or for the {E286 appointment} of a receiver or similar officer for it, {E287 makes} an {E288 assignment} for the benefit of all or substantially all of its creditors, or {E289 enters} into an agreement for the {E290 composition} , {E291 extension} , or {E292 readjustment} of all or substantially all of its obligations or admits of its general inability to {E293 pay} its debts as they {E294 become} due. D. In the event of {E295 termination} under this section, VIP will {E296 give} BNL, at its {E297 request} and E - 9 {E298 direction} , such copies of BNL data {E299 maintained} on the VIIP system in a format and in a manner as {E300 designated} by BNL. BNL shall {E301 pay} a fee to VIP for {E302 preparing} such data. Such fee shall be $100 per hour for programming time and $150 per hour computer processing time. 8. INDEMNIFICATION. A.
Response: E2 COREFERENCE E12
E2 COREFERENCE E53
E2 COREFERENCE E88
E2 COREFERENCE E118
E2 COREFERENCE E133
E2 COREFERENCE E223
E2 COREFERENCE E227
E2 COREFERENCE E234
E2 COREFERENCE E238
E3 COREFERENCE E4
E3 COREFERENCE E14
E3 COREFERENCE E49
E3 COREFERENCE E52
E3 COREFERENCE E89
E3 COREFERENCE E132
E3 COREFERENCE E224
E3 COREFERENCE E228
E3 COREFERENCE E235
E3 COREFERENCE E239
E45 COREFERENCE E46
E73 COREFERENCE E78
E92 COREFERENCE E96
E116 COREFERENCE E120
E116 COREFERENCE E122
E116 COREFERENCE E124
E116 COREFERENCE E131
E143 COREFERENCE E144
E187 COREFERENCE E202
E193 COREFERENCE E195
E257 COREFERENCE E259
E262 COREFERENCE E269
E262 COREFERENCE E273
E274 COREFERENCE E279


**Example 2:**
Text:VIRTUAL ITEM PROCESSING SYSTEMS, INC. 2525 Northwest Expressway, #105 Oklahoma City, Oklahoma 73112 OUTSOURCING AGREEMENT BETWEEN VIRTUAL ITEM PROCESSING SYSTEMS, INC. And BROKERS NATIONAL LIFE ASSURANCE COMPANY E - 4 OUTSOURCING AGREEMENT This Outsourcing Agreement (" Agreement") is {E0 executed} as of this 1 st day of May 2006 , by and between Virtual Item Processing Systems, Inc. ("VIP"), with its principal place of office at 2525 NW Expressway, Suite 105 Oklahoma City, Oklahoma 73112, and Brokers National Life Assurance Company ("BNL"), with its principal place of office at 7010 Hwy. 71 W., Suite 100, Austin, Texas 78735. WHEREAS, VIP is {E1 engaged} in the business of {E2 providing} Electronic Data Processing {E3 services} ("EDP {E4 Services} ") and related {E5 consultation} and {E6 services} to insurance companies pursuant to computer software systems {E7 developed} and {E8 owned} by VIP , (the "VIP System"); WHEREAS, BNL is an insurance company {E9 domiciled} in the State of Arkansas and {E10 licensed} to {E11 do} business in numerous additional states; and WHEREAS, VIP desires to {E12 provide} EDP Services to BNL; and WHEREAS, BNL desires to {E13 obtain} EDP {E14 services} from VIP for the {E15 processing} and {E16 administration} of its insurance policies; NOW, THEREFORE, in consideration of the above premises and in consideration of other good and valuable consideration, the {E17 receipt} and sufficiency is hereby {E18 acknowledged} , the parties {E19 agree} as follows : 1. PURCHASE OF EQUIPMENT. BNL at its expense shall {E20 obtain} , {E21 install} , {E22 maintain} and {E23 upgrade} as necessary any and all hardware, software, data and telephone lines, other communications equipment and any other equipment (hereinafter collectively referred to as the "Equipment") which it {E24 determines} is necessary to {E25 allow} it to {E26 use} and {E27 access} the VIP System pursuant to the terms of this Agreement. Such Equipment shall be fully compatible with the VIP System. VIP will {E28 provide} BNL such information as is reasonably necessary to {E29 allow} BNL to {E30 acquire} all such Equipment which {E31 meets} the requirements of this paragraph. If {E32 requested} by BNL and at BNL's expense, VIP shall {E33 inspect} all such Equipment and {E34 acknowledge} its compatibility in writing prior to its use with the VIP System. 2. VIP's EQUIPMENT AND SERVICES. A. During the term of this Agreement, VIP shall {E35 provide} BNL such access as necessary to the VIP System to {E36 allow} BNL to {E37 attach} one data communication line and up to seventy (70) addressable data communications devices to {E38 said} VIP System. Should BNL {E39 desire} to {E40 attach} additional communication lines or additional communication devices to the VIP System, BNL shall {E41 pay} to {E42 VIP} the additional fees set forth in paragraph 5(F) of this Agreement. B. VIP, at its sole discretion and expense, may, but is not {E43 obligated} to, {E44 make} appropriate {E45 enhancements} to the VIP System. Any such {E46 enhancements} shall be {E47 deemed} to be {E48 included} in the EDP {E49 Services} and VIP System to be {E50 provided} to BNL, whether {E51 developed} by VIP before or during the time when {E52 services} are to be {E53 provided} by VIP pursuant to this Agreement. During the term of this Agreement, VIP shall be responsible at its expense for the proper {E54 maintenance} and {E55 documentation} of the VIP System. 3 .SCHEDULED AND {E56 UNSCHEDULED} DOWN TIME. BNL {E57 acknowledges} that there will be {E58 scheduled} downtime for the routine preventive {E59 maintenance} of VIP's System {E60 performed} by either VIP or its vendors. VIP shall {E61 give} BNL reasonable advance notice of all such scheduled downtime. BNL further {E62 acknowledges} that there will also be {E63 unscheduled} down-time that might {E64 occur} as a result of electrical power {E65 failures} and equipment {E66 failures} and other {E67 acts} outside of the control of VIP as {E68 contemplated} in paragraph 16(J). In the event that any such down-time {E69 extends} for more than two (2) consecutive working days , VIP, at its expense, will {E70 make} available to BNL access to a backup facility {E71 designated} by VIP for the {E72 continued} {E73 processing} of BNL's business. To {E74 ensure} that a backup facility will be available in E - 5 case of such a {E75 failure} , VIP will {E76 maintain} disaster and/or business interruption insurance adequate to {E77 establish} alternate site {E78 processing} , as {E79 provided} for in paragraph 12(A) of this Agreement. 4. INCLUDED SERVICES IN THE VIP SYSTEM. It is {E80 agreed} and {E81 understood} by BNL that: A. It has {E82 reviewed} and {E83 inspected} the VIP System {E84 existing} as of the Effective Date of this Agreement, which VIP System {E85 includes} (i) a New Business System, (ii) a Policy Administration System, (iii) an Agency Administration System, (iv) a Financial Administration System. (v) a Claims System, (vi) a Vendor Provider System, (vii) a Transaction Tracking System and (viii) a Mail Tracking System; B. Such VIP System as {E86 identified} in paragraph 4(A) is adequate to {E87 meet} the needs of BNL; C. VIP shall {E88 provide} EDP {E89 Services} to BNL for such Initial Policies and policies identical thereto and {E90 renewals} thereof by the {E91 use} of such VIP System {E92 existing} as of the Effective Date of this Agreement, except as such VIP System may be {E93 modified} from time to time by VIP , at the discretion of VIP; D. BNL has {E94 reviewed} the security system (Security System") {E95 included} in the VIP System {E96 existing} as of the Effective Date of this Agreement; E. BNL {E97 acknowledges} and {E98 agrees} that such Security System is adequate to {E99 protect} the confidential information and data of BNL {E100 processed} by the VIP System; F. BNL, throughout the term of this Agreement, shall be solely responsible for {E101 choosing} , {E102 implementing} and {E103 utilizing} any or all of such of the security measures and protections {E104 offered} by {E105 said} Security System for the {E106 use} of or access to the VIP System by any of its officers, directors, shareholders, employees and agents; G. VIP shall not have any duty to either {E107 monitor} or {E108 enforce} such security {E109 measures} and {E110 protections} {E111 chosen} , {E112 implemented} or {E113 utilized} by BNL; H. E - 6 BNL shall be solely responsible for any {E114 acts} or {E115 omissions} of any of its officers, directors, shareholders, employees and agents; I. Notwithstanding anything to the contrary in this Agreement, VIP at any time during the term of this Agreement may {E116 change} the platform upon which the VIP System is {E117 operated} and through which the EDP Services are {E118 provided} to BNL by VIP under this Agreement. Before VIP shall {E119 make} such platform {E120 change} VIP shall {E121 give} BNL prior reasonable written notice of such {E122 change} , and VIP's warranties under this Agreement shall {E123 continue} notwithstanding such {E124 change} and VIP {E125 agrees} to {E126 pay} any cost {E127 created} for or {E128 imposed} on BNL for equipment, {E129 training} or similar matters {E130 arising} from such {E131 change} . 5. PAYMENTS TO VIP. A. For EDP {E132 Services} {E133 provided} pursuant to this Agreement, BNL will {E134 pay} to {E135 VIP} the {E136 charges} set forth in the Payment Schedule {E137 attached} hereto as Schedule B: {E138 provided} however and notwithstanding anything to the contrary herein. The minimum monthly fee shall not be less than five thousand dollars ($5,000) per month (as applicable, "Minimum Fee"). B. For any additional VIP {E139 Services} {E140 provided} hereunder, BNL will {E141 pay} to {E142 VIP} the {E143 charges} set {E144 charges} set forth in the Payment Schedule {E145 attached} hereto as Schedule A. C. The fees due hereunder are subject to the {E146 following} provisions: 1. The fee for each new policy {E147 submitted} into the VIP System is set forth in Schedule B. 2. VIP will {E148 process} all policies that have thirteen (13) or more months {E149 expired} from their original policy date at the annual rates set forth in Schedule B. with a separate fee for each renewal base policy and each rider, for each plan, {E150 prorated} to the actual number of months each policy is {E151 represented} to be in force on the VIP System. Such representation of "policy status" {E152 includes} the "grace period" and "Late payment {E153 offer} " that each policy may {E154 enjoy} and in which case {E155 exceeds} a time frame not {E156 bound} by each policy's actual {E157 paid} for period. The payment amount for each group of policies in a rate category will be {E158 calculated} by {E159 determining} the actual number of policies and riders in force that are {E160 included} in the rate category, as set forth in Schedule B at the end of each calendar month and then {E161 multiplying} the number of policies by the base policy renewal amount and the number of riders by the rider renewal amount then {E162 adding} the totals together and {E163 dividing} the {E164 resulting} amount by twelve (12). The amounts {E165 calculated} for all rate categories are {E166 added} together and this amount is the fee payable in advance at the beginning of the month . 3. VIP shall not be {E167 obligated} to {E168 process} any {E169 amended} policies or new products that E - 7 are {E170 written} or {E171 acquired} by BNL unless and until the parties hereto have mutually {E172 executed} a {E173 written} addendum to this Agreement {E174 modifying} Schedule B to {E175 include} the fees for any such products. D. Any sum due VIP hereunder for which a time for {E176 payment} is not otherwise specified will be due and payable within ten (10) days after the date of the postmark for an invoice therefor from VIP. If BNL {E177 fails} to {E178 pay} any amount due within ten (10) days from the date of the postmark for the invoice, late {E179 charges} of 1-1/2% per month , or the maximum amount allowable by law, whichever is less, shall also {E180 become} payable by BNL to VIP. E. In addition to the communication line and devices which BNL is {E181 authorized} to {E182 attach} to the VIP System pursuant to paragraph 2 of this Agreement, BNL may, for the monthly fee(s) hereinafter set forth, {E183 attach} additional communication lines or the {E184 following} devices to the VIP System. The monthly fee(s) for such additional lines) or devices is as follows : 1. each communication line and {E185 adapter} $200, 2. each visual station whether CRT, PC or similar device $25, 3. each addressable printer under 299 lines per minute ("LPM") $25 4. each addressable printer over 299 LPM $150. F. There are certain other expenses which are directly {E186 related} to VIP's {E187 performance} of this Agreement that are directly billable by VIP and payable by BNL. The purpose and intent of this provision is not to {E188 describe} all {E189 contemplated} {E190 charges} {E191 covered} by this provision, but rather to {E192 identify} some of the {E193 charges} that may {E194 fall} into this category . Such {E195 charges} {E196 include} but are not limited to the following : 1. Cost of all business forms, continuous or non-continuous {E197 used} by BNL; 2. All telephone {E198 calls} {E199 initiated} on behalf of BNL business and operations; 3. All travel, food and lodging expenses {E200 incurred} by VIP personnel {E201 related} to the {E202 performance} of this Agreement, subject to BNL's prior {E203 written} approval; 4. All postage and shipping expenses for materials {E204 used} by BNL; 5. All expenses {E205 incurred} for computer output micro-film "COM" which is {E206 contracted} by VIP with a service bureau independent of VIP , subject to BNL' s prior {E207 written} approval; 6. Any other {E208 charges} directly {E209 related} to BNL ' {E210 use} or benefit of the VIP System pursuant to this Agreement is subject to BNL ' prior {E211 written} approval. G. All sums due under this Agreement are payable in U.S. dollars. 6. PROPRIETARY AND RELATED RIGHTS. A. CLIENT DATA. Any original documents or files {E212 provided} to VIP hereunder by BNL ("BNL Data") are and shall {E213 remain} BNL's property and, upon the {E214 termination} of this Agreement for any reason, such BNL Data will be {E215 returned} to BNL by VIP, subject to E - 8 the terms hereof. Subject to paragraphs 4(F) and (G), VIP {E216 agrees} to {E217 make} the same effort to {E218 safeguard} such BNL Data as it {E219 does} in {E220 protecting} its own proprietary information. BNL Data will not be {E221 utilized} by VIP for any purpose other than those purposes {E222 related} to {E223 rendering} EDP {E224 Services} to BNL under this Agreement, nor will BNL Data or any part thereof be {E225 disclosed} to third parties by VIP , its employees or agents except for purposes {E226 related} to VIP's {E227 rendering} ofEDP {E228 Services} to BNL under this Agreement or as {E229 required} by law, regulation, or {E230 order} of a court or regulatory agency or other authority {E231 having} jurisdiction thereover. Notwithstanding the foregoing, VIP shall have the right to {E232 retain} in its possession all work papers and files {E233 prepared} by it in {E234 performance} of EDP {E235 Services} hereunder which may {E236 include} necessary copies of BNL Data. VIP shall have access to BNL Data, at reasonable times, during the term of this Agreement and thereafter for purposes {E237 related} to VIP's ' {E238 rendering} of EDP {E239 Services} to BNL pursuant to this Agreement, or as {E240 required} by law, regulation or {E241 order} of a court or regulatory agency or other authority {E242 having} jurisdiction thereover. Notwithstanding the foregoing, the confidentiality obligations set forth in this paragraph will not {E243 apply} to any information which (i) is or {E244 becomes} publicly available without {E245 breach} of this Agreement, (ii) is independently {E246 developed} by VIP outside the scope of this Agreement and without reference to the confidential information {E247 received} under this Agreement, or (iii) is rightfully {E248 obtained} by VIP from third parties which are not {E249 obligated} to {E250 protect} its confidentiality. 7. TERMINATION FOR CAUSE. This Agreement may be {E251 terminated} by the non- breaching party upon any of the {E252 following} {E253 events} : A. In the event that BNL {E254 fails} to {E255 pay} any sums of money due to VIP hereunder and does not {E256 cure} such {E257 default} within thirty (30) days after {E258 receipt} of written notice of such {E259 nonpayment} from VIP , {E260 provided} that if BNL {E261 notifies} VIP in writing that BNL {E262 disputes} a billing and BNL {E263 pays} any undisputed portion of such billing VIP shall not {E264 institute} formal {E265 proceedings} by {E266 arbitration} or judicial {E267 review} or {E268 terminate} this Agreement with respect to such {E269 disputed} billing until after VIP has {E270 afforded} BNL an opportunity for a {E271 meeting} to {E272 discuss} such {E273 dispute} . B. In the event that a party hereto {E274 breaches} any of the material terms, covenants or conditions of this Agreement (other than a {E275 breach} under paragraph (A) above) and {E276 fails} to {E277 cure} the same within thirty (30) days after {E278 receipt} of written notice of such {E279 breach} from the non-breaching party. C. In the event that a party hereto {E280 becomes} or is {E281 declared} insolvent or bankrupt, is the subject of any {E282 proceedings} {E283 relating} to its {E284 liquidation} , {E285 insolvency} or for the {E286 appointment} of a receiver or similar officer for it, {E287 makes} an {E288 assignment} for the benefit of all or substantially all of its creditors, or {E289 enters} into an agreement for the {E290 composition} , {E291 extension} , or {E292 readjustment} of all or substantially all of its obligations or admits of its general inability to {E293 pay} its debts as they {E294 become} due. D. In the event of {E295 termination} under this section, VIP will {E296 give} BNL, at its {E297 request} and E - 9 {E298 direction} , such copies of BNL data {E299 maintained} on the VIIP system in a format and in a manner as {E300 designated} by BNL. BNL shall {E301 pay} a fee to VIP for {E302 preparing} such data. Such fee shall be $100 per hour for programming time and $150 per hour computer processing time. 8. INDEMNIFICATION. A.
Response: E2 COREFERENCE E12
E2 COREFERENCE E53
E2 COREFERENCE E88
E2 COREFERENCE E118
E2 COREFERENCE E133
E2 COREFERENCE E223
E2 COREFERENCE E227
E2 COREFERENCE E234
E2 COREFERENCE E238
E3 COREFERENCE E4
E3 COREFERENCE E14
E3 COREFERENCE E49
E3 COREFERENCE E52
E3 COREFERENCE E89
E3 COREFERENCE E132
E3 COREFERENCE E224
E3 COREFERENCE E228
E3 COREFERENCE E235
E3 COREFERENCE E239
E45 COREFERENCE E46
E73 COREFERENCE E78
E92 COREFERENCE E96
E116 COREFERENCE E120
E116 COREFERENCE E122
E116 COREFERENCE E124
E116 COREFERENCE E131
E143 COREFERENCE E144
E187 COREFERENCE E202
E193 COREFERENCE E195
E257 COREFERENCE E259
E262 COREFERENCE E269
E262 COREFERENCE E273
E274 COREFERENCE E279


Now analyze the following text:"""+f"\nText: {prompt} \nResponse:")