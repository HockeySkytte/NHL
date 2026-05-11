# PUCKPEDIA INC. API SERVICE AGREEMENT

**Draft for Discussion**

This API Service Agreement (the "Agreement") is entered into as of May 6, 2026 (the "Effective Date") by and between **PuckPedia Inc.** ("PuckPedia" or "Provider") and **Lars Sunesen Skytte** ("Subscriber" or "Licensee").

This draft is intended to reflect the parties' business understanding regarding access to PuckPedia's API feeds, approved website uses, attribution requirements, rate limits, and a more balanced allocation of contractual risk.

## Chapter 1. Recitals

WHEREAS, PuckPedia operates the website located at https://PuckPedia.com and makes certain hockey-related content and data available through its website and application programming interface;

WHEREAS, Subscriber desires to access certain API feeds made available by PuckPedia for use on Subscriber's approved website properties and for Subscriber's internal business purposes, subject to the terms of this Agreement; and

WHEREAS, the parties desire to define the permitted scope of use, attribution requirements, API access conditions, and respective rights and obligations relating to such access.

NOW, THEREFORE, in consideration of the mutual covenants and agreements set forth herein, the parties agree as follows:

## Chapter 2. Definitions

"API" means the application programming interface made available by PuckPedia under this Agreement, together with any related documentation, credentials, technical specifications, and supporting materials made available by PuckPedia to Subscriber.

"Approved Uses" means the authorized uses of the API and Content expressly set forth in Chapter 4 and Exhibit B.

"Content" means the data, feeds, lineup information, contract-related information, salary cap information, player information, and other information made available by PuckPedia to Subscriber through the API.

"Derived Analytics" means calculations, models, rankings, projections, comparisons, metrics, summaries, visualizations, and other analytical outputs created by Subscriber using the Content, so long as such outputs do not expose or permit reconstruction of the API, the raw feed, or the Content in its entirety.

"Internal Use" means Subscriber's internal business use of the API and Content in connection with Subscriber's website operations, analytics, product development, editorial workflows, and subscription features, in each case consistent with this Agreement.

"PuckPedia Website" means the website operated by PuckPedia and accessible at https://PuckPedia.com.

"Subscriber's Website" means the website(s), subdomains, subdirectories, mobile applications, and access-controlled subscription areas owned or operated by Subscriber and approved by PuckPedia in writing, including the uses described in Exhibit B.

## Chapter 3. Service

3.1 Subject to the terms of this Agreement, PuckPedia shall provide Subscriber access to the API feeds described in Exhibit A.

3.2 PuckPedia may make reasonable updates to the API, including technical modifications, format changes, and maintenance changes, provided that such changes do not materially reduce the functionality of the API without reasonable prior notice where practicable.

3.3 PuckPedia shall use commercially reasonable efforts to maintain the availability of the API, subject to scheduled maintenance, outages, third-party dependencies, and force majeure events.

## Chapter 4. License; Purpose; Approved Uses

4.1 Subject to the terms of this Agreement, PuckPedia grants Subscriber a limited, non-exclusive, non-transferable, non-sublicensable, revocable license during the Term to access and use the API and Content on Subscriber's Website and for Internal Use.

4.2 Subscriber may use the API and Content for the Approved Uses described in this Chapter 4 and Exhibit B.

4.3 The Approved Uses include the following:

1. **Public player pages.** Subscriber may display selected player contract information and related contract data points on publicly accessible skater pages and goaltender pages on Subscriber's Website.

2. **Public team pages.** Subscriber may display team cap hit and related salary cap information on publicly accessible team pages on Subscriber's Website, including aggregate or segmented cap hit values by forwards, defensemen, and goaltenders.

3. **Game projection calculations.** Subscriber may use lineup data supplied through the API as an input into Subscriber's internal and public-facing game projection calculations, models, and related analysis.

4. **Subscription roster builder.** Subscriber may use lineup data and team cap hit data supplied through the API within a subscription-required roster builder feature on Subscriber's Website, including the display of selected lineup, roster, and cap-related data points to authenticated subscribers.

5. **Value analysis and derived analytics.** Subscriber may use contract information, cap hit information, and lineup data supplied through the API to create and display Derived Analytics on Subscriber's Website, including:

   a. team-level "Value per Million" analysis for forwards, defensemen, and goaltenders; and

   b. league-level analysis by contract type, including ELC, RFA, and UFA, and by player age.

4.4 Notwithstanding any general restriction elsewhere in this Agreement, the Approved Uses expressly authorized in this Chapter 4 and Exhibit B shall be deemed permitted uses and shall not, by themselves, constitute unauthorized publication, disclosure to a third party, or operation of a competing service, provided that Subscriber complies with the display, attribution, anti-republication, and API access restrictions in this Agreement.

## Chapter 5. Use Restrictions

5.1 Subscriber shall not:

1. share, resell, sublicense, assign, lease, or otherwise provide the API, API credentials, or raw API feed to any third party;

2. expose API endpoints, credentials, tokens, or non-public API documentation in client-side code or to the public;

3. publish or distribute the Content in a format that allows the raw Content or a substantially complete feed to be copied, downloaded, exported, scraped, or reconstructed in its entirety by third parties;

4. exceed the API rate limits set forth in Exhibit A or otherwise use the API in a manner that unreasonably interferes with PuckPedia's systems;

5. use the API or Content to create, support, or maintain a standalone public-facing database or service that substantially replicates PuckPedia's contract and cap database in a manner competitive with PuckPedia; or

6. remove any required attribution to PuckPedia when displaying Content-derived data points.

5.2 For clarity, Subscriber may display selected data points, limited values, summaries, and Derived Analytics consistent with the Approved Uses, so long as such display does not expose the raw API feed or permit reconstruction of the Content in its entirety.

## Chapter 6. Attribution

6.1 When displaying Content or data points sourced from the API, Subscriber shall provide attribution to PuckPedia in a commercially reasonable manner, including a reference to PuckPedia as the source and a direct link to https://PuckPedia.com where reasonably practicable.

6.2 The parties acknowledge that attribution may be provided by page-level source attribution, tooltip attribution, footnote attribution, or other reasonably visible attribution suitable to the relevant product surface.

## Chapter 7. API Access; Rate Limits; Storage

7.1 Subscriber may access the API up to the rate limit specified in Exhibit A.

7.2 Subscriber may store, cache, normalize, and process Content internally during the Term for performance, analytics, modeling, and product functionality, provided that such storage is reasonably related to the Approved Uses and does not result in unauthorized redistribution of the Content.

7.3 Subscriber may perform a nightly pull and may also perform reasonable update calls during the day, including lineup refreshes, so long as Subscriber remains within the agreed rate limits.

## Chapter 8. Fees; Term; Renewal

8.1 The initial Term shall commence on the Effective Date and continue through April 30, 2027.

8.2 This Agreement shall automatically renew for successive twelve-month periods unless either party gives written notice of non-renewal at least thirty (30) days before the end of the then-current Term.

8.3 Subscriber shall pay the fees set forth in Exhibit A.

## Chapter 9. Termination; Suspension; Effect of Termination

9.1 Either party may terminate this Agreement for material breach by the other party if such breach remains uncured for thirty (30) days after written notice describing the breach in reasonable detail.

9.2 PuckPedia may suspend API access immediately upon written notice if Subscriber's use presents an actual security risk, involves unauthorized disclosure of API credentials, or materially exceeds the agreed rate limits in a manner that threatens PuckPedia's systems; provided that PuckPedia shall restore access promptly once the applicable issue is cured.

9.3 Upon termination or expiration, Subscriber shall cease new use of the API and shall cease public display of Content-based data obtained solely from the API, except that Subscriber may retain:

1. archival backups;

2. logs and compliance records;

3. internal analytics records; and

4. Derived Analytics and models that do not expose or permit reconstruction of the raw Content.

## Chapter 10. Representations and Warranties

10.1 Each party represents and warrants that:

1. it has full right, power, and authority to enter into and perform this Agreement; and

2. its execution and performance of this Agreement will not violate any other agreement binding upon it.

10.2 PuckPedia represents and warrants that:

1. it has the right to provide the API and license the Content as contemplated by this Agreement;

2. the API and Content, when used by Subscriber as expressly authorized under this Agreement, will not knowingly infringe any third-party intellectual property rights; and

3. it will use commercially reasonable efforts to maintain the availability of the API, subject to maintenance, outages, third-party dependencies, and force majeure events.

10.3 Subscriber represents and warrants that:

1. it will use the API and Content in accordance with this Agreement and applicable law; and

2. it will not knowingly use the API in a manner that exposes API credentials or raw feed access to third parties.

## Chapter 11. Disclaimer

EXCEPT AS EXPRESSLY SET FORTH IN THIS AGREEMENT, THE API AND CONTENT ARE PROVIDED "AS IS" AND "AS AVAILABLE," AND EACH PARTY DISCLAIMS ALL IMPLIED WARRANTIES, INCLUDING IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. PUCKPEDIA DOES NOT WARRANT THAT THE API WILL BE UNINTERRUPTED OR ERROR-FREE, BUT SHALL USE COMMERCIALLY REASONABLE EFFORTS TO ADDRESS MATERIAL API FAILURES.

## Chapter 12. Limitation of Liability

12.1 EXCEPT FOR EXCLUDED CLAIMS, NEITHER PARTY SHALL BE LIABLE TO THE OTHER FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, EXEMPLARY, OR PUNITIVE DAMAGES, OR FOR ANY LOST PROFITS, LOST REVENUE, OR LOSS OF GOODWILL, ARISING OUT OF OR RELATING TO THIS AGREEMENT.

12.2 EXCEPT FOR EXCLUDED CLAIMS, EACH PARTY'S AGGREGATE LIABILITY ARISING OUT OF OR RELATING TO THIS AGREEMENT SHALL NOT EXCEED THE GREATER OF:

1. the fees paid or payable under this Agreement during the twelve (12) months preceding the event giving rise to the claim; or

2. Ten Thousand U.S. Dollars (US $10,000).

12.3 "Excluded Claims" means:

1. a party's fraud, willful misconduct, or gross negligence;

2. Subscriber's unauthorized disclosure of API credentials or raw API access;

3. either party's breach of Chapter 14 (Confidential Information);

4. amounts expressly due under this Agreement; and

5. either party's indemnification obligations under Chapter 13.

## Chapter 13. Indemnification

13.1 PuckPedia shall indemnify, defend, and hold harmless Subscriber from and against any third-party claim alleging that Subscriber's authorized use of the API or Content under this Agreement infringes such third party's intellectual property rights, except to the extent the claim arises from:

1. Subscriber's modification of the Content;

2. Subscriber's combination of the Content with other content or systems not supplied by PuckPedia, where the claim would not have arisen but for such combination; or

3. Subscriber's use outside the scope of this Agreement.

13.2 Subscriber shall indemnify, defend, and hold harmless PuckPedia from and against any third-party claim arising out of:

1. Subscriber's use of the API or Content in breach of this Agreement;

2. Subscriber's violation of applicable law; or

3. content independently supplied by Subscriber on Subscriber's Website.

13.3 The indemnified party shall promptly notify the indemnifying party of any claim, reasonably cooperate in the defense, and permit the indemnifying party to control the defense and settlement, provided that no settlement admitting fault or imposing obligations on the indemnified party may be entered without the indemnified party's prior written consent, not to be unreasonably withheld.

## Chapter 14. Confidential Information

14.1 "Confidential Information" means non-public information disclosed by one party to the other that is designated as confidential or that reasonably should be understood to be confidential given the nature of the information and the circumstances of disclosure, including API credentials, non-public API documentation, non-public endpoint specifications, technical implementation details, and non-public business information.

14.2 Confidential Information does not include:

1. information that is or becomes public through no breach of this Agreement;

2. information independently developed without use of the disclosing party's Confidential Information;

3. information lawfully received from a third party without confidentiality restriction; or

4. data points and information already publicly displayed on the PuckPedia Website, except for the non-public API access methods, credentials, feed structure, and documentation used to deliver such information.

14.3 Each party shall protect the other party's Confidential Information using reasonable care and shall use such Confidential Information only as necessary to perform under this Agreement.

14.4 If disclosure is required by law, the receiving party shall, where legally permitted, provide prompt notice and reasonable cooperation.

## Chapter 15. Intellectual Property; Derived Works

15.1 PuckPedia retains all right, title, and interest in and to the API, API documentation, PuckPedia branding, and the Content as delivered through the API, subject to the license granted in this Agreement.

15.2 Subscriber retains all right, title, and interest in and to Subscriber's Website, Subscriber's branding, and Subscriber's Derived Analytics, models, projections, rankings, visualizations, and product features, provided that such assets do not expose or reproduce the raw API or Content in violation of this Agreement.

15.3 Nothing in this Agreement transfers ownership of either party's intellectual property to the other.

## Chapter 16. Equitable Relief; Dispute Resolution; General

16.1 Each party acknowledges that unauthorized disclosure of API credentials, misuse of Confidential Information, or unauthorized republication of raw API Content may cause irreparable harm for which monetary damages may be insufficient. Accordingly, either party may seek injunctive or equitable relief in addition to any other available remedies.

16.2 This Agreement shall be governed by the laws of the State of California, without regard to conflicts-of-law principles.

16.3 Except for claims seeking injunctive or equitable relief, any dispute arising out of or relating to this Agreement shall be resolved by binding arbitration administered by JAMS before a single arbitrator.

16.4 The arbitration shall be conducted remotely by video conference unless the parties agree otherwise.

16.5 Each party shall bear its own fees and costs, except that the arbitrator may award reasonable attorneys' fees and costs to the prevailing party where appropriate.

16.6 This Agreement, including Exhibit A and Exhibit B, constitutes the entire agreement between the parties with respect to its subject matter and supersedes all prior or contemporaneous understandings relating thereto.

16.7 In the event of any conflict between a general restriction in this Agreement and an expressly authorized Approved Use in Exhibit B, the Approved Use shall control.

16.8 Any amendment to this Agreement must be in writing and signed by both parties.

16.9 Neither party may assign this Agreement without the prior written consent of the other party, except in connection with a merger, acquisition, or sale of substantially all of its assets.

16.10 If any provision of this Agreement is held invalid or unenforceable, the remainder of the Agreement shall remain in full force and effect.

16.11 This Agreement may be executed in counterparts and by electronic signature, each of which shall be deemed an original and all of which together shall constitute one instrument.

## Chapter 17. Notices

If to Subscriber:

Lars Sunesen Skytte  
Finsensgade 56, St. Tv.  
8200 Aarhus N, Denmark  
lars.sunesen.skytte@gmail.com

If to Provider:

PuckPedia Inc.  
159 Gariepy Crescent  
Edmonton, AB T6M 1B5  
Attn: Hart Levine  
Hart@puckpedia.com

## Chapter 18. Signatures

**SUBSCRIBER**

By: ________________________________

Name: Lars Sunesen Skytte

Title: ______________________________

Date: _______________________________

**PUCKPEDIA INC.**

By: ________________________________

Name: Hart Levine

Title: CEO

Date: _______________________________

## Exhibit A. API Service; Update Frequency; Rate Limits; Fees

1. **API Service.** PuckPedia shall provide the following JSON feeds:

   a. Lineups API

   b. Players API

2. **Update Frequency.** New contract-related updates are generally added promptly after public announcement and, absent unusual circumstances, by end of day.

3. **API Call Frequency.** Subscriber may call the API up to five (5) requests per sixty (60) seconds. PuckPedia recommends a nightly pull between 1:00 a.m. and 3:00 a.m. Pacific Time. Subscriber may also make reasonable daytime refreshes, including lineup-related refreshes, provided Subscriber remains within the agreed rate limit.

4. **Fees.**

   Annual Subscription Fee: US $7,500  
   Less Discount: (US $7,140)  
   Net Service Fee: US $360

5. **Attribution.** Subscriber shall provide attribution to PuckPedia consistent with Chapter 6 of this Agreement.

## Exhibit B. Approved Uses Schedule

PuckPedia approves the following uses of the API and Content by Subscriber on Subscriber's Website and for Internal Use during the Term:

1. Display of selected player contract information and related contract data points on public skater and goaltender pages.

2. Display of team cap hit information on public team pages, including aggregate cap hit and segmented cap hit values by forwards, defensemen, and goaltenders.

3. Use of lineup data as an input into Subscriber's game projection calculations, models, and related public-facing or internal analysis.

4. Use of lineup data and team cap hit data within Subscriber's subscription-required roster builder feature.

5. Use of contract information, cap hit information, and lineup data to create and display Derived Analytics, including:

   a. team-level value-per-million analysis for forwards, defensemen, and goaltenders; and

   b. league-level analysis by contract type, including ELC, RFA, and UFA, and by age.

6. Display of approved Content and Derived Analytics on both public pages and access-controlled subscription pages, provided that Subscriber does not expose the raw API feed or permit copying, downloading, or reconstruction of the Content in its entirety.