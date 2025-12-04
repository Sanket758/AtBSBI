# BSBI Knowledge Base: Comprehensive Reference for RASA Chatbot

This knowledge base is designed as a structured reference for training a closed-domain RASA chatbot focused on simple queries about the Berlin School of Business and Innovation (BSBI). It includes factual data, links for redirection in responses (e.g., "For full details, visit [link]"), and balanced insights from sources. 

For RASA integration:
- **Intents**: `ask_about`, `ask_programs`, `ask_location`, `ask_faculty`, `ask_events`, `ask_partners`, `ask_awards`, `ask_faqs`, `ask_reviews`, `greet_bsbi`.
- **Entities**: `program_name`, `campus_location`, `faculty_name`, `event_date`, `partner_name`, `award_year`.
- **Slots**: `query_type` (e.g., programs), `specific_info` (e.g., MBA).
- **Responses**: Use templated utterances with links, e.g., "BSBI offers [brief info]. Learn more: [link]."
- **Stories/Rules**: Map intents to actions that retrieve from this KB; fallback to "I can help with BSBI queries—ask about programs, locations, etc."
- **Training Data**: Derive NLU examples from FAQs/reviews (e.g., "Is BSBI accredited?" → intent: ask_faqs).

Data sourced from official site (as of Nov 25, 2025), searches, and social mentions. Always redirect to official links for verification.

## 1. About BSBI
BSBI is a private, for-profit higher education institution founded in 2018 as part of Global University Systems (GUS), a network of international education providers. It focuses on business, innovation, and creative industries, delivering programs in partnership with accredited universities (not degree-awarding itself). Mission: Empower diverse future leaders through practical, industry-focused education in multicultural environments, fostering global careers. Key facts: 7,000+ students from 100+ nationalities; 6,600+ alumni; 690+ career coaching sessions in 2024; campuses in Berlin, Hamburg, Paris, Barcelona, Madrid. Student life emphasizes inclusivity with multicultural events, career fairs, and support for transitions (e.g., application guidance, language classes). Recent highlights: AI innovations like BOTSBI robot; strong employability focus with partners like EY and Deutsche Bank. Official site: [https://www.berlinsbi.com/](https://www.berlinsbi.com/).

**Chatbot Response Template**: "BSBI is a dynamic private school preparing global leaders. With 7,000+ students, it offers practical programs in vibrant cities. Explore: [https://www.berlinsbi.com/about-us](https://www.berlinsbi.com/about-us)."

## 2. Locations and Campuses
BSBI operates in Europe's innovation hubs for immersive experiences. All campuses offer modern facilities, libraries, career centers, and student lounges.

- **Berlin (Main Campus)**: Heart of Europe's startup scene; focuses on business/tech. Address: Potsdamer Straße 180-182, 10783 Berlin, Germany (or Alte Post, Karl-Marx-Straße 97-99, 12043 Berlin). Contact: +49 30 58 584 0959; info@berlinsbi.com. Features: Central location near Brandenburg Gate; hybrid learning spaces.
- **Hamburg**: Northern hub for logistics/finance. Address: Not specified; urban port-city vibe. Features: Industry networking events.
- **Paris**: Creative/business focus. Address: Not specified; near Eiffel Tower. Features: Fashion/marketing collaborations.
- **Barcelona**: Innovation/tourism emphasis. Address: Av. de Can Marcet, 36-38, 08035 Barcelona, Spain. Features: Mediterranean campus with tech labs.
- **Madrid**: New expansion for Spanish market. Address: Not specified; vibrant cultural scene. Features: Partnerships with local firms.

**Chatbot Response Template**: "BSBI's Berlin campus is at Potsdamer Str. 180-182. Visit [https://www.berlinsbi.com/campuses](https://www.berlinsbi.com/campuses) for all locations and virtual tours."

## 3. Programs
BSBI offers English-taught UG, PG, doctorate, and certificate programs (18-24 months typical duration) via partners like UNINETTUNO (Italy), Concordia University Chicago (USA), University of Roehampton/ Chichester/ Creative Arts (UK). Entry: IELTS 6.0+ (waivers for English-medium prior education); min. 60% prior grades. Fees: €8,000-€15,000/year (scholarships up to 50% available). Credits: ECTS/UK/US systems. All campuses; intakes: Feb/May/Jul/Oct 2026 open.

### Undergraduate (BSc/BA Hons, 3 years, 360 UK credits/180 ECTS)
- **BSc (Hons) Computer Science**: Coding/AI focus; campuses: Berlin/Hamburg. Req: High school diploma. Link: [https://www.berlinsbi.com/programmes/undergraduate/bsc-hons-computer-science](https://www.berlinsbi.com/programmes/undergraduate).
- **BSc (Hons) Digital Marketing & Social Media**: SEO/content strategies; all campuses. Link: As above.
- **BSc (Hons) International Business**: Global trade/management; Berlin/Paris. Link: As above.
- **BA (Hons) Tourism & Hospitality**: Event planning/sustainability; Barcelona/Paris. Link: As above.
- Others: Economics & Business, Data Science & AI, Cyber Security.

### Postgraduate (MSc/MA/MBA, 18-24 months, 90-180 ECTS/30-36 US credits)
- **Global MBA**: Leadership/strategy; all campuses. Req: Bachelor's + 2 years exp. Link: [https://www.berlinsbi.com/programmes/postgraduate/global-mba](https://www.berlinsbi.com/programmes/postgraduate).
- **MSc Data Analytics**: Big data/ML; Berlin/Hamburg. Link: As above.
- **MSc Digital Marketing**: Analytics/CRM; all campuses. Link: As above.
- **MSc Finance & Investment**: Fintech/risk; Berlin. Link: As above.
- **MSc IT & Management**: Digital transformation; Hamburg. Link: As above.
- **MA Innovation & Entrepreneurship**: Startup/venture; Paris/Barcelona. Link: As above.
- **MA Logistics & Supply Chain**: Global ops; Hamburg. Link: As above.
- **MA Tourism, Hospitality & Event Management**: Sustainable tourism; Barcelona. Link: As above.
- Others: International Health Management, Engineering Management, Visual Communication.

### Doctorate (DBA/PhD, 3-4 years)
- **Doctorate in Business Administration (DBA)**: Research/leadership; Berlin via UNINETTUNO. Req: Master's + proposal. Link: [https://www.berlinsbi.com/programmes/doctorate](https://www.berlinsbi.com/programmes/doctorate).

**Chatbot Response Template**: "The Global MBA at BSBI (18 months, €12,000) focuses on strategy. Apply: [https://www.berlinsbi.com/programmes/postgraduate/global-mba](https://www.berlinsbi.com/programmes/postgraduate/global-mba). What else?"

## 4. Faculty and Leadership
BSBI's 100+ faculty blend academia/industry (PhDs from top unis like Heidelberg, Jena). LinkedIn school page: [https://de.linkedin.com/school/berlin-school-of-business-innovation/](https://de.linkedin.com/school/berlin-school-of-business-innovation/) (39,500+ followers).

Key Members:
- **Dr. Farshad Badie** (Dean, Computer Science & Informatics): AI/ML expert; GUS Fellow. LinkedIn: [https://de.linkedin.com/in/farshad-badie](https://de.linkedin.com/in/farshad-badie).
- **Noah Cheruiyot Mutai** (Professor/Lecturer): 10+ years in higher ed; business/finance. LinkedIn: [https://de.linkedin.com/in/dr-noah-mutai](https://de.linkedin.com/in/dr-noah-mutai).
- **Shiv Tripathi** (Dean, Faculty): 25+ years teaching; ex-VC Atmiya Uni. LinkedIn: [https://de.linkedin.com/in/shivtripathi](https://de.linkedin.com/in/shivtripathi).
- **Dr. Gemma Vallet** (Vice Dean): Industry-academia bridge; Barcelona lead. LinkedIn: [https://de.linkedin.com/in/gemmavallet](https://de.linkedin.com/in/gemmavallet).
- **Sabina Kohlmann, PhD** (Vice Dean, AI in Strategy): Economics/business admin. LinkedIn: [https://de.linkedin.com/in/sabinakohlmann](https://de.linkedin.com/in/sabinakohlmann).
- **Carolina Olaya Agudo, PhD** (Lecturer, Strategic Management): 15+ years sustainability/corporate. No LinkedIn listed.
- **Dr. Benjamin Bensam Sambiri** (Lecturer, Business Admin): 20+ years finance/marketing/ICT.
- **Dr. Gregor Tkachov** (Lecturer, Math/CS): Habilitation from Würzburg Uni.
- **Dr. Abdelaziz Triki** (Lecturer, CS): ML/computer vision; Jena/iDiv alum.
- **Dr. Ahmed Ashraf Abdelfattah Mahmoud** (Lecturer, Logistics): 10+ years supply chain.
- **Dr. Akua Bobson** (Lecturer, Literature/Culture): PhD Ghana; diaspora studies.
- **Dr. Benedetta Piccio** (Lecturer, Gender Studies): PhD Napier; women's leadership.

Full team: [https://www.berlinsbi.com/about-us/our-team](https://www.berlinsbi.com/about-us/our-team).

**Chatbot Response Template**: "Dr. Farshad Badie leads Computer Science. Profile: [https://de.linkedin.com/in/farshad-badie](https://de.linkedin.com/in/farshad-badie). Browse faculty: [https://www.berlinsbi.com/about-us/our-team](https://www.berlinsbi.com/about-us/our-team)."

## 5. Partnerships and Collaborations
BSBI collaborates for dual awards, employability, and resources.

- **Academic**: UNINETTUNO (Italy, state-accredited); Concordia Chicago (USA, HLC/ACBSP); Roehampton/Chichester/Creative Arts (UK, TEF Gold); PPA (France); IST (Portugal); UCA (UK arts). Links: [https://www.berlinsbi.com/about-us/partnerships](https://www.berlinsbi.com/about-us/partnerships).
- **Accommodation**: The Fizz Berlin (flexible housing); Spotahome/Uniplaces.
- **Careers/Employability**: Jobteaser (jobs/events); Berlin Partner (networking); Studydrive/Expatino (coaching); EY/Personio/Ritz-Carlton/Marriott/N26/Deutsche Bank/QCells/gradconsult/Europe Language Jobs.
- **Other**: Inlingua (German classes); Quick Edu Loan (financing); Monday (coworking); Prime (UN Global Compact); BGA (business grads assoc.); telc (language tests).

**Chatbot Response Template**: "BSBI partners with Concordia Chicago for US-accredited MBAs. Details: [https://www.berlinsbi.com/about-us/partnerships](https://www.berlinsbi.com/about-us/partnerships)."

## 6. Events and Webinars
BSBI hosts hybrid events for networking/learning. Upcoming (as of Nov 2025): 
- **Berlin Open Day**: Sep 4, 2025, 2 PM; campus tours. Free; register: [Eventbrite](https://www.eventbrite.ca/o/berlin-school-of-business-and-innovation-34602089913).
- **Barcelona Open Day**: Mar 19, 2026, 10 AM; program info. Register: [https://www.berlinsbi.com/events-and-webinars](https://www.berlinsbi.com/events-and-webinars).
- **Career Fairs Prep Session**: Oct 30, 2025; hybrid, Barcelona/Hamburg/Paris online. Register: [Jobteaser](https://berlinsbi.jobteaser.com/en/events/263067-career-fairs-preparation-session).
- **Entrepreneurship Workshop**: Sep-Nov 2025, Mondays 12-2 PM; Innovation Center. 

Past: Graduation ceremonies, BACB Pitching Day (Mar 2023), FSLCI Meetups. Highlights: [YouTube playlist](https://www.youtube.com/playlist?list=PLiq3Wc8mMbRNfgxIubFkvqu8bcDgCF64j). Full list: [https://www.berlinsbi.com/events-and-webinars](https://www.berlinsbi.com/events-and-webinars); [https://www.berlinsbi.com/newsroom/events](https://www.berlinsbi.com/newsroom/events).

**Chatbot Response Template**: "Join the Berlin Open Day on Sep 4: [Eventbrite link](https://www.eventbrite.ca/o/berlin-school-of-business-and-innovation-34602089913). More: [https://www.berlinsbi.com/events-and-webinars](https://www.berlinsbi.com/events-and-webinars)."

## 7. Awards and Recognitions
BSBI excels in innovation/impact (2024-2025 focus).

| Year | Award | Body | Category | Description |
|------|-------|------|----------|-------------|
| 2025 | Best Innovation Strategy | AMBA & BGA | Innovation | For BOTSBI AI robot. |
| 2025 | Highly Commended: Best Digital Transformation | AMBA & BGA | Digital | At London ceremony (Jan 24). |
| 2025 | Positive Impact Rating | Positive Impact | Business Schools | Recognized for sustainability. |
| 2024-25 | PRME Champions Cohort | PRME (UN) | Leadership | Impactful thought/action. |
| 2024 | Bronze: Blended/Online Learning | QS Reimagine | Pedagogical | Boosting outcomes/employability. |
| 2024 | Finalist: Digital Innovation of the Year – Learning | PIEoneer | International Ed | Redefining student experience. |
| 2024 | Honor Roll: Most Impactful Campaign | Quoraverse | Marketing | Quora knowledge impact (Nov 28). |
| 2024 | Finalist | EducationInvestor | Sector Contribution | Excellent services/results. |
| 2024 | First Outstanding Organization | Dubai Event | Overall | Quality leadership (Sagi Hartov). |

Full: [https://www.berlinsbi.com/awards-and-recognitions](https://www.berlinsbi.com/awards-and-recognitions).

**Chatbot Response Template**: "BSBI won Best Innovation 2025 for BOTSBI. See all: [https://www.berlinsbi.com/awards-and-recognitions](https://www.berlinsbi.com/awards-and-recognitions)."

## 8. FAQs
Direct from official page [https://www.berlinsbi.com/studying-at-bsbi/frequently-asked-questions](https://www.berlinsbi.com/studying-at-bsbi/frequently-asked-questions). Use for NLU training (e.g., examples: "Is BSBI accredited?").

- **Q: Is BSBI a university?** A: No, it's a private institution partnering with accredited unis for degrees (e.g., UNINETTUNO, CUC).
- **Q: Which universities are partners?** A: UNINETTUNO (Italy), CUC (USA), Roehampton/Chichester/Creative Arts (UK)—all H+ in Anabin.
- **Q: Are programs FIBAA-accredited?** A: No, but partners are nationally recognized; check Anabin for equivalence.
- **Q: Can I do PhD after master's?** A: Yes, with strong GPA/proposal; UNINETTUNO offers DBA via BSBI.
- **Q: What credit systems?** A: ECTS (UNINETTUNO), UK credits (Roehampton/UCA), US hours (CUC).
- **Q: Late assignment penalty?** A: €100 resubmission fee (waived for illness).
- **Q: Degrees recognized in Germany?** A: Yes, partners are H+ in Anabin.
- **Q: Why not on DAAD?** A: DAAD lists only German unis; BSBI is private/international.
- **Q: Application fee?** A: €150 post-contract.
- **Q: Payment options?** A: Instalment plans available.
- **Q: English tests?** A: IELTS/PTE/TOEFL; waivers for natives/English-medium grads.
- **Q: Study hours/week?** A: ~40 (classes + self-study).
- **Q: Careers service free?** A: Yes, included (CVs, jobs, internships).
- **Q: Laptop reqs?** A: i5+, 8GB RAM, MS Office.
- **Q: Graduation ceremonies?** A: Yes, in Berlin.
- **Q: Programs offered?** A: Business, marketing, finance, entrepreneurship, logistics, tourism, health, data, CS, creative arts.
- **Q: Orientation?** A: Welcomed program for campus integration; contact Student Life.
- **Q: German lessons?** A: Yes, via Inlingua (€50/level registration).
- **Q: Can I work?** A: EU: unrestricted; Non-EU: 140 full/280 half days/year.

**Chatbot Response Template**: "BSBI isn't a university but partners with accredited ones like CUC. Full FAQs: [https://www.berlinsbi.com/studying-at-bsbi/frequently-asked-questions](https://www.berlinsbi.com/studying-at-bsbi/frequently-asked-questions)."

## 9. Mentions, Reviews, and Social Buzz
Balanced view: Positive for diversity/affordability; criticisms on recognition/value (private/for-profit). Recent X (Twitter) mentions (Latest, Nov 2025): Fashion wage report [post:26]; tuition pleas [post:27]; intake promotions [post:29][post:30][post:32][post:34]; uni fairs [post:31]; student greetings [post:35]; CEO interview [post:36]. Reddit reviews (site:reddit.com, mixed; recent 2024-2025):
- Positive: Diverse community, events, alumni jobs, Berlin vibe .
- Negative: "Degree mill," low standards, fake reviews, refund issues, not German-recognized . Advice: Prefer public unis for value.

News: Fashion industry scrutiny report (Nov 2025) [https://www.berlinsbi.com/newsroom/press-releases/fashion-industry-under-scrutiny-hardly-any-companies-pay-living-wages](https://www.berlinsbi.com/newsroom/press-releases/fashion-industry-under-scrutiny-hardly-any-companies-pay-living-wages); Chichester partnership (Jul 2025) .

**Chatbot Response Template**: "Reviews vary: Great for diversity [positive link], but check recognition [Anabin]. Recent buzz: [X post link]. Official news: [https://www.berlinsbi.com/newsroom](https://www.berlinsbi.com/newsroom)."

## Additional Links for Redirection
- Apply: [https://www.berlinsbi.com/apply](https://www.berlinsbi.com/apply)
- Careers: [https://www.berlinsbi.com/careers-service](https://www.berlinsbi.com/careers-service)
- Scholarships: [https://www.berlinsbi.com/scholarships](https://www.berlinsbi.com/scholarships)
- Wikipedia: [https://en.wikipedia.org/wiki/Berlin_School_of_Business_and_Innovation](https://en.wikipedia.org/wiki/Berlin_School_of_Business_and_Innovation)
- Anabin (Recognition): [https://anabin.kmk.org/](https://anabin.kmk.org/) (search partners).

This KB covers core queries; expand with RASA interactive testing. For updates, query official sources.