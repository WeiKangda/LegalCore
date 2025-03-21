import os
import sys
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForCausalLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from post_processing.utils import load_jsonl, append_to_jsonl, process_coreference, create_coreference_clusters, replace_elements_with_mentions, mentions_to_clusters
from pre_processing.utils import generate_paths
from eval import save_metrics_to_file, calculate_micro_macro_muc, calculate_micro_macro_b3, calculate_micro_macro_ceaf_e, calculate_micro_macro_blanc
import api_utils

def generate_response(model,is_commercial, tokenizer, prompt, inference_mode):
    if inference_mode=="zero_shot":
        msgs = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
            "role": "user",
            "content": r"""Please analyze the following text to detect all coreference relations among events. 
Two events have a coreference relation if they refer to the same event in space and time. 
Coreference relation is symmetrical (i.e., non-directional): If A coreferences B, then B coreferences A. 
It is also transitive: If A coreferences B and B coreferences C, then A coreferences C. 

Each event is uniquely identified in the text by an identifier in the format {E## trigger_word}. 
For example:
  - {E01 discovered}
  - {E02 collaborated}
  - {E03 agreed}

Response Format:
List all coreference relations strictly following this format:
  E01 COREFERENCE E03
  E02 COREFERENCE E05

IMPORTANT:
- Use exactly the same event identifiers as in the text.
- Do not change the format of the event IDs (always use E##).
- If there are multiple coreference relations, list each on a new line.
- If no coreference relation is detected, return "None" (do not add any explanation).
- If examples are provided, they are for illustration only. Do not copy the event identifiers from the examples. Use only the event identifiers found in the provided text.                      
"""+f"Text: {str(prompt)}\nResponse:"
            }
        ]

    elif inference_mode == "one_shot":
        msgs = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": r"""Please analyze the following text to detect all coreference relations among events. 
Two events have a coreference relation if they refer to the same event in space and time. 
Coreference relation is symmetrical (i.e., non-directional): If A coreferences B, then B coreferences A. 
It is also transitive: If A coreferences B and B coreferences C, then A coreferences C. 

Each event is uniquely identified in the text by an identifier in the format {E## trigger_word}. 
For example:
  - {E01 discovered}
  - {E02 collaborated}
  - {E03 agreed}

Response Format:
List all coreference relations strictly following this format:
  E01 COREFERENCE E03
  E02 COREFERENCE E05

IMPORTANT:
- Use exactly the same event identifiers as in the text.
- Do not change the format of the event IDs (always use E##).
- If there are multiple coreference relations, list each on a new line.
- If no coreference relation is detected, return "None" (do not add any explanation).
- If examples are provided, they are for illustration only. Do not copy the event identifiers from the examples. Use only the event identifiers found in the provided text.                                                  
**Example:**
Text:Exhibit 10.1 {E0 Redactions} with respect to certain portions hereof {E1 denoted} with "***" COLLABORATION AGREEMENT This Collaboration Agreement (the "Agreement") is {E2 made} as of April 14th, 2020 (the "Effective Date ") by and between Anixa Biosciences, Inc., a Delaware corporation, {E3 located} at 3150 Almaden Expressway, Suite 250, San Jose, CA 95118, U.S.A. ("Anixa"), and OntoChem GmbH, a German limited liability company, {E4 located} at BlÃ¼cherstr. 24, D-06120 Halle (Saale), Germany ("OntoChem"). Anixa and OntoChem are {E5 referred} to herein individually as a "Party" and collectively as the "Parties." WHEREAS, the Parties {E6 wish} to {E7 collaborate} in the {E8 discovery} and {E9 development} of novel drug candidates for the {E10 treatment} of COVID-19 in accordance with the terms and conditions of this Agreement. NOW, THEREFORE, in consideration of the premises and the mutual promises set forth in this Agreement, and other good and valuable consideration, the {E11 receipt} and sufficiency of which are hereby {E12 acknowledged} , the Parties {E13 agree} as follows : 1. {E14 Defined} Terms. 1.1 "Affiliate" means, with respect to a Party, any entity directly or indirectly controlled by, controlling or under common control with such Party. For purposes of this definition, "control" means (a) ownership of fifty percent (50%) (or such lesser percentage which is the maximum allowed to be owned by a foreign entity or investor in a particular jurisdiction) or more of the outstanding voting stock or other ownership interest of an entity, or (b) possession of the power to (i) elect, appoint, direct or remove fifty percent (50%) or more of the members of the board of directors or other governing body of an entity or (ii) otherwise direct or cause the direction of the {E15 management} or policies of an entity by contract or otherwise. 1.2 "Hit Compound" means any chemical entity that is determined in performing the Research Plan to meet the Hit Criteria. 1.3 "Hit Criteria" means the criteria identified as "Hit Criteria" as set forth in the Research Plan. 1.4 " {E16 Invention} " means any {E17 invention} , know-how, data, {E18 discovery} or proprietary information, whether or not patentable, that is made or generated solely by the Representatives of Anixa or OntoChem or jointly by the Representatives of Anixa and OntoChem in performing the Research Plan, including all intellectual property rights in the foregoing. 1.5 "Representative" means, with respect to a Party, an officer, director, employee, agent or permitted subcontractor of such Party. 1.6 "Research Plan" means the research plan attached hereto as Exhibit A. 1 1.7 "SAR" means the relationship between the chemical or three-dimensional structure of a compound and its biological activity, and includes the {E19 determination} of the chemical groups responsible for evoking a target biological {E20 effect} . 1.8 "Target" means: (a) any protease of any coronavirus, including Mpro; (b) the Nsp15-pRB ribonuclease protein- protein {E21 interaction} ; (c) all mutants and variants of any molecule or component referenced in clauses (a) or (b); and (d) all truncated forms (including fragments) of any molecule or component referenced in clauses (a) or (b) or mutant or variant referenced in clause (c). 1.9 "Variant" means, with respect to any Hit Compound: (a) all compounds within the genus of compounds to which such Hit Compound would belong under United States patent laws as referenced in the Selection Notice (as defined below); and (b) any base form, metabolite, ester, salt form, racemate, stereoisomer, polymorph, hydrate, anhydride or solvate of such Hit Compound or any other compound described in clause (a) (in the case of this clause (b), without regard to whether such compound is referenced in the Selection Notice). 2. Research Program. 2.1 Performance. The Parties will diligently {E22 perform} their respective {E23 activities} set forth in the Research Plan (such {E24 activities} , collectively, the "Research {E25 Program} ") in accordance with the timelines set forth therein, with the objective of {E26 identifying} Hit Compounds and Lead Scaffolds that {E27 modulate} the applicable Target. Without {E28 limiting} the foregoing, OntoChem will (a) {E29 provide} all deliverables set forth in the Research Plan (each, a "Deliverable") and (b) {E30 obtain} any {E31 authorizations} , {E32 approvals} and licenses {E33 required} for {E34 performance} of the Research Plan. If any terms set forth in the Research Plan conflict with the terms set forth in this Agreement, the terms of this Agreement will {E35 control} unless expressly {E36 indicated} to the contrary in the Research Plan. The Research Plan may not be {E37 amended} without the prior {E38 written} consent of both Parties. If, from time to time, the Parties {E39 desire} to {E40 expand} the scope of the Research Program, then they will {E41 negotiate} in good faith a potential {E42 amendment} of the Research Plan in regard to such {E43 expanded} scope, on commercially reasonable terms, but neither Party will be {E44 obligated} to {E45 enter} into any such {E46 amendment} . 2.2 Weekly Updates. OntoChem will {E47 provide} Anixa with weekly (or more frequently as {E48 requested} ) {E49 updates} {E50 regarding} its {E51 progress} under the Research {E52 Program} via teleconference, videoconference or e-mail, and the Parties will {E53 make} appropriate personnel available in a timely manner to {E54 discuss} and {E55 provide} {E56 feedback} in regard to such {E57 updates} . 2.3 Delivery of Data. In conjunction with each weekly update {E58 described} in Section 2.2, OntoChem will {E59 deliver} to Anixa all data {E60 generated} under the Research Plan since the {E61 preceding} {E62 update} . In addition, Anixa will have the right to reasonably {E63 request} additional information {E64 relating} to such data, and OntoChem will {E65 respond} to such {E66 requests} promptly with any such additional information in its possession or control, {E67 provided} that, for clarity, OntoChem will not be {E68 required} to {E69 perform} any new or additional {E70 research} in order to {E71 generate} any such additional information. 2 2.4 Selection of Lead Scaffolds. Within one year {E72 following} {E73 completion} of all {E74 activities} under the Research Plan (the "Selection Deadline"), Anixa, in good faith {E75 consultation} with OntoChem, will have the right to {E76 select} up to two hundred (200) Hit Compounds (each, a "Selected Hit Compound"), by {E77 providing} OntoChem with written notice of such {E78 Selected} Hit Compound(s) (the "Selection Notice"), and each Selected Hit Compound, along with all Variants of such Selected Hit Compound {E79 referenced} in the Selection Notice, is hereby {E80 designated} as a "Lead Scaffold" under this Agreement. {E81 Commencing} upon {E82 selection} of a Selected Hit Compound, Anixa (itself and through its Affiliates and designees) will have sole authority over and control of the further {E83 development} , {E84 manufacture} , and {E85 commercialization} of the corresponding Lead Scaffold and any product candidate or product {E86 incorporating} a compound from such Lead Scaffold. {E87 Following} the Selection Deadline, Anixa will have no further rights with respect to any Hit Compound that is not a Selected Hit Compound or {E88 included} within a Lead Scaffold (each, a "Rejected Hit Compound"), {E89 provided} that, during the period of two (2) years following the Selection Deadline, neither OntoChem nor any of its Affiliates will {E90 use} or {E91 disclose} to any third party any Rejected Hit Compound or any Variant thereof, {E92 including} the identity, structure or SAR information of any such compound, for application as anti-viral agents or protease inhibitors, for purposes of {E93 modulating} any Target or for {E94 treatment} of virus- {E95 related} conditions. In case OntoChem {E96 finds} a novel and unexpected antiviral {E97 use} of those {E98 Rejected} {E99 Hit} Compounds during this 2-years period , it will {E100 notify} Anixa about these {E101 findings} and Anixa has the right of first {E102 negotiation} during a period of 6 months after this {E103 notification} . If Anixa {E104 decides} to not {E105 license} those {E106 uses} or compounds for this novel antiviral use, OntoChem is free to {E107 develop} those molecules further as its own intellectual property without any further restrictions. 2.5 Subcontractors. OntoChem may {E108 engage} one or more subcontractors to {E109 perform} its {E110 activities} under the Research Plan with the prior {E111 written} approval of Anixa and {E112 provided} that, with respect to any such subcontractor, OntoChem will (a) be responsible and liable for the {E113 performance} of such subcontractor and (b) {E114 enter} into a {E115 written} agreement (i) consistent with terms and conditions of this Agreement, {E116 including} with respect to confidentiality and intellectual property, and (ii) {E117 prohibiting} such subcontractor from further subcontracting. For clarity, vendors where commercial building blocks or compounds will be {E118 purchased} are nor {E119 regarded} as subcontractors. 2.6 Target Exclusivity. During the term of this Agreement, except in the {E120 performance} of its obligations or {E121 exercise} of its rights under this Agreement, neither OntoChem nor any of its Affiliates will {E122 discover} , {E123 research} , {E124 develop} , {E125 manufacture} or {E126 commercialize} any compound or product {E127 directed} to any Target, either independently or for or in collaboration with a third party ( {E128 including} the grant of a license to any third party), or have any of the foregoing {E129 activities} {E130 performed} on behalf of OntoChem or any of its Affiliates by a third party. For clarity, the foregoing {E131 includes} the {E132 screening} ( {E133 including} via computational methods) of any compound library or virtual compound library against any Target. 2.7 Records. Each Party will {E134 maintain} complete and accurate records of all {E135 activities} {E136 performed} by or on behalf of such Party under the Research Program and all {E137 Inventions} {E138 made} or {E139 generated} by or on behalf of such Party in the {E140 performance} of the Research Program. Such records will be in sufficient detail and in good scientific manner appropriate for patent and regulatory purposes. Each Party will {E141 provide} the other Party with the right to {E142 inspect} such records, and upon {E143 request} will {E144 provide} copies of all such records, to the extent reasonably {E145 required} for the {E146 exercise} or {E147 performance} of such other Party's rights or obligations under this Agreement, {E148 provided} that any information {E149 disclosed} under this Section 2.7 will be subject to the terms and conditions of Section 5. Each Party will {E150 retain} such records for at least three (3) years following {E151 expiration} or {E152 termination} of this Agreement or such longer period as may be {E153 required} by applicable law or regulation. 3 2.8 Debarment. Each Party hereby {E154 represents} and warrants to the other Party that neither it nor any of its Affiliates or personnel has been {E155 debarred} under any health care laws or regulations and that, to its knowledge, no {E156 investigations} , {E157 claims} or {E158 proceedings} with respect to {E159 debarment} are {E160 pending} or {E161 threatened} against such Party or any of its Affiliates or personnel. Neither Party nor any of its Affiliates will {E162 use} in any capacity, in connection with the Research Program, any person or entity who has been {E163 debarred} . Each Party {E164 agrees} and {E165 undertakes} to promptly {E166 notify} the other Party if such Party or any of its Affiliates or personnel {E167 becomes} {E168 debarred} or {E169 proceedings} have been {E170 initiated} against any of them with respect to {E171 debarment} , whether such {E172 debarment} or {E173 initiation} of {E174 proceedings} {E175 occurs} during or after the term of this Agreement. 3. Financial Terms. 3.1 Research Program Payments. In consideration for OntoChem's {E176 performance} of its {E177 activities} under the Research Plan, Anixa will: (a) {E178 pay} OntoChem 100,002 Euros in six (6) equal installments as follows : (i) 16,667 Euros within five (5) days after the Effective Date ; and (ii) five (5) installments in the amount of 16,667 Euros on each one-month anniversary of the Effective Date , except that the last such {E179 payment} will be due within thirty (30) days after {E180 completion} of all {E181 activities} under the Research Plan; and (b) {E182 reimburse} OntoChem for its out-of-pocket expenses {E183 incurred} in {E184 performing} the Research Plan on a pass- through basis without mark-up, within thirty (30) days after {E185 delivery} of an invoice therefore ( {E186 including} reasonable {E187 supporting} documentation), {E188 provided} that Anixa has {E189 approved} such expenses in advance and in writing ( including in regard to the {E190 selection} of specific Hit Compounds to be {E191 synthesized} and {E192 analyzed} in biological assays). It is {E193 estimated} that OntoChem's out-of-pocket expenses under the Research Plan will {E194 include} 110,000 Euros payable to Tube Pharmaceuticals GmbH as a subcontractor of OntoChem, subject to Section 2.5. (c) High-throughput screening compounds OntoChem will forward a commercial {E195 proposal} to {E196 acquire} these compounds at the sole discretion of Anixa. Both parties will {E197 agree} on payment conditions. (d) Extra custom synthesis OntoChem will forward a commercial {E198 proposal} to have {E199 synthesized} these compounds at the sole discretion of Anixa. Both parties will {E200 agree} on payment conditions. (e) Biological testing OntoChem will forward a commercial {E201 proposal} to have biologically {E202 test} these compounds at the sole discretion of Anixa. Both parties will {E203 agree} on payment conditions. 3.2 Lead Scaffold Payments. For each Lead Scaffold {E204 selected} by Anixa, Anixa will {E205 pay} OntoChem an annual fee of 10,000 U.S. Dollars, payable within thirty (30) days following each anniversary of the date of the Selection Notice, until five (5) years after the first commercial {E206 sale} of the first product {E207 incorporating} a compound from such Lead Scaffold, subject to Section 4.3 with respect to any Terminated Scaffold (as {E208 defined} below). 3.3 Milestone Payment. Anixa will {E209 pay} OntoChem a one-time milestone {E210 payment} of 300,000 U.S. Dollars within thirty (30) days following the {E211 dosing} of the first patient in the first human clinical trial for the first product {E212 incorporating} a compound from a Lead Scaffold. 4 3.4 Payment Terms. {E213 Payments} to OntoChem will be {E214 made} by check or by wire transfer of immediately available funds to such bank account as {E215 designated} in writing by OntoChem from time to time. Taxes (and any penalties and interest thereon) {E216 imposed} on any {E217 payment} {E218 made} by Anixa to OntoChem will be the responsibility of OntoChem. The fees for the respective bank {E219 transfers} will be {E220 borne} by Anixa. 3.5 Financial Records. OntoChem will {E221 maintain} complete and accurate books and accounting records {E222 related} to all out-of-pocket expenses {E223 incurred} in {E224 performing} the Research Plan. These records will be available for {E225 inspection} during regular business hours upon reasonable {E226 notice} by Anixa, or its duly {E227 authorized} representative, at Anixa's expense, for three (3) years following the end of the calendar year in which such expenses are {E228 invoiced} . If it is {E229 determined} that Anixa has {E230 overpaid} for any expenses {E231 passed} through by OntoChem under this Agreement, OntoChem will promptly {E232 reimburse} Anixa for the amount of such {E233 overpayment} and, if such {E234 overpayment} {E235 represents} more than five percent (5%) of the corresponding amount due, OntoChem will {E236 pay} Anixa's reasonable fees and expenses {E237 incurred} in connection with such {E238 inspection} . 4. Term and Termination. 4.1 Term . Unless earlier {E239 terminated} in accordance with Section 4.2 or 4.3, this Agreement will be in effect from the Effective Date until {E240 completion} of the Research Program. 4.2 Termination by Anixa. This Agreement may be {E241 terminated} by Anixa, without cause, upon at least thirty (30) days {E242 written} notice to OntoChem. 4.3 Termination of Lead Scaffolds.
Response: E8 COREFERENCE E18
E16 COREFERENCE E137
E22 COREFERENCE E34
E22 COREFERENCE E140
E22 COREFERENCE E176
E22 COREFERENCE E184
E22 COREFERENCE E224
E23 COREFERENCE E24
E23 COREFERENCE E25
E23 COREFERENCE E74
E23 COREFERENCE E110
E23 COREFERENCE E129
E23 COREFERENCE E135
E23 COREFERENCE E177
E23 COREFERENCE E181
E40 COREFERENCE E43
E42 COREFERENCE E46
E63 COREFERENCE E66
E78 COREFERENCE E82
E96 COREFERENCE E101
E100 COREFERENCE E103
E155 COREFERENCE E168
E155 COREFERENCE E172
E169 COREFERENCE E174
E170 COREFERENCE E173
E209 COREFERENCE E210
E213 COREFERENCE E217
E230 COREFERENCE E233
E230 COREFERENCE E234


Now analyze the following text:"""+f"\nText: {prompt} \nResponse:"
            }
        ]
    elif inference_mode == "two_shot":
        msgs = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": r"""Please analyze the following text to detect all coreference relations among events. 
Two events have a coreference relation if they refer to the same event in space and time. 
Coreference relation is symmetrical (i.e., non-directional): If A coreferences B, then B coreferences A. 
It is also transitive: If A coreferences B and B coreferences C, then A coreferences C. 

Each event is uniquely identified in the text by an identifier in the format {E## trigger_word}. 
For example:
  - {E01 discovered}
  - {E02 collaborated}
  - {E03 agreed}

Response Format:
List all coreference relations strictly following this format:
  E01 COREFERENCE E03
  E02 COREFERENCE E05

IMPORTANT:
- Use exactly the same event identifiers as in the text.
- Do not change the format of the event IDs (always use E##).
- If there are multiple coreference relations, list each on a new line.
- If no coreference relation is detected, return "None" (do not add any explanation).
- If examples are provided, they are for illustration only. Do not copy the event identifiers from the examples. Use only the event identifiers found in the provided text.                      
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


Now analyze the following text:"""+f"\nText: {prompt} \nResponse:"
            }
        ]
    print(msgs[1]["content"])
    if is_commercial:
        content = model.eval_call(msgs, debug=False)
        response = model.resp_parse(content)[0]
    else:
        input_ids = tokenizer.apply_chat_template(
            msgs,
            padding=True,
            return_tensors="pt",
        )
        # Generate text from the model
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=1024,
        )
        prompt_length = input_ids.shape[1]
        response = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
    return response

def event_coreference(model,is_commercial, tokenizer, data, inference_mode):
    result = {"id": data["id"]}
    text = data['singleton_text']
    mention_list = data["events"]
    response = generate_response(model,is_commercial, tokenizer, text, inference_mode)
    print("-----------event_coreference response--------------\n", response)
    result["response"] = response
    coreference_tuples = process_coreference(response)
    clusters = create_coreference_clusters(coreference_tuples)
    result["clusters"] = replace_elements_with_mentions(clusters, mention_list)

    return result

def event_coreference_end2end(model,is_commercial, tokenizer, data, inference_mode):
    result = {"id": data["id"]}
    text = data['text_with_predicted_event']
    mention_list = data["events"]
    response = generate_response(model,is_commercial, tokenizer, text, inference_mode)
    print("-----------event_coreference_end2end response--------------\n", response)
    result["response"]=response
    coreference_tuples = process_coreference(response)
    print(coreference_tuples)
    clusters = create_coreference_clusters(coreference_tuples)
    print(clusters)

    result["clusters"] = replace_elements_with_mentions(clusters, mention_list)

    return result

def run_event_coreference(model_name,is_commercial,data_path,output_path,inference_mode):
    print(model_name)
    if is_commercial:
        tokenizer = None
        model = api_utils.GPT(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id, device_map="auto", trust_remote_code=True)
        model.eval()

    all_data = load_jsonl(data_path)
    # Set the output and result file path for event detection
    task_name = "coreference"
    base_dir = output_path
    output_file, final_result_file = generate_paths(base_dir, task_name, model_name, inference_mode)

    all_predicted = [] 
    all_gold = []
    for data in tqdm(all_data):
        result = event_coreference(model,is_commercial, tokenizer, data, inference_mode)
        append_to_jsonl(output_file, result)
        all_predicted.append(result["clusters"])
        all_gold.append(mentions_to_clusters(data["events"]))
        print("Gold mentions:" + str(mentions_to_clusters(data["events"])))
        print("Predicted mentions:" + str(result["clusters"]))
        print("########################")

    final_result = {}
    muc = calculate_micro_macro_muc(all_gold, all_predicted)
    print("MUC:" + str(muc))
    b3 = calculate_micro_macro_b3(all_gold, all_predicted)
    print("B^3:" + str(b3))
    ceaf_e = calculate_micro_macro_ceaf_e(all_gold, all_predicted)
    print("CEAF_e:" + str(ceaf_e))
    blanc = calculate_micro_macro_blanc(all_gold, all_predicted)
    print("BLANC:" + str(blanc))

    final_result["MUC"] = muc
    final_result["B^3"] = b3
    final_result["CEAF_e"] = ceaf_e
    final_result["BLANC"] = blanc
    #print(final_result)
    save_metrics_to_file(final_result, final_result_file)