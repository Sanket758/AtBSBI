from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SessionStarted, ActionExecuted, SlotSet
from database_logger import init_db, log_session_start, update_session_end

# Initialize DB when actions server starts
init_db()

class ActionSessionStart(Action):
    def name(self) -> Text:
        return "action_session_start"

    async def run(self, dispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        session_id = tracker.sender_id
        log_session_start(session_id)
        
        # Required to actually start the session in Rasa
        events = [SessionStarted()] 
        events.append(ActionExecuted("action_listen"))
        return events

class ActionGetProgramDetails(Action):

    def name(self) -> Text:
        return "action_get_program_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        response = None
        # Get the program name from the slot (detected by NLU)
        program_name = tracker.get_slot("program").replace("&", "and")
        print(f"DEBUG: Action Triggered. Slot 'program' value: {program_name}")
        
        if not program_name:
            dispatcher.utter_message(text="Which program are you interested in? We have MBA, AI, Tourism, and more.")
            return []
        
        # THE KNOWLEDGE BASE (Your table converted to a Python Dictionary)
        # Using partial matching keys for robustness
        knowledge_base = {
            "economics": {
                "name": "BA in Economics and Business Administration (UNINETTUNO)",
                "duration": "3 years",
                "campus": "Berlin, Paris",
                "desc": "European bachelor's degree with 180 ECTS.",
                "link": "https://www.berlinsbi.com/programmes/undergraduate/ba-in-economics-and-business-administration"
            },
            "tourism": {
                "name": "BA (Hons) Tourism and Hospitality Management",
                "duration": "3 years",
                "campus": "Berlin",
                "desc": "Develops critical thinking for tourism & hospitality sectors.",
                "link": "https://www.berlinsbi.com/programmes/undergraduate/ba-tourism-and-hospitality-management"
            },
            "animation": {
                "name": "BA (Hons) Animation",
                "duration": "3 years",
                "campus": "Hamburg",
                "desc": "Combines theory & practice to create short films.",
                "link": "https://www.berlinsbi.com/programmes/undergraduate/ba-hons-animation"
            },
            "graphic": {
                "name": "BA (Hons) Graphic Design (UCA)",
                "duration": "3 years",
                "campus": "Hamburg",
                "desc": "Builds powerful software skills and creative thinking.",
                "link": "https://www.berlinsbi.com/programmes/undergraduate/ba-hons-graphic-design"
            },
            "psychology": {
                "name": "BSc (Hons) Psychology",
                "duration": "3 years",
                "campus": "Berlin",
                "desc": "Scientific exploration of mind, brain & behaviour.",
                "link": "https://www.berlinsbi.com/programmes/undergraduate/bsc-hons-psychology"
            },
            "mba": {
                "name": "Global MBA",
                "duration": "18 months",
                "campus": "Berlin, Paris",
                "desc": "Covers marketing, finance, operations, and leadership.",
                "link": "https://www.berlinsbi.com/programmes/postgraduate/global-mba-uca"
            },
            "finance": {
                "name": "MSc Finance & Investment",
                "duration": "18 months",
                "campus": "Berlin",
                "desc": "Advanced skills to manage finance sector challenges.",
                "link": "https://www.berlinsbi.com/programmes/postgraduate/msc-finance-investment"
            },
            "blockchain": {
                "name": "Certificate in Blockchain and Cryptocurrency",
                "duration": "Short course",
                "campus": "Berlin",
                "desc": "Foundational understanding of blockchain and crypto.",
                "link": "https://www.berlinsbi.com/programmes/certificate-programmes/blockchain-and-cryptocurrency"
            },
            "computer_science": {
                "name": "BSc (Hons) Computer Science",
                "duration": "3 years",
                "campus": "Berlin, Hamburg",
                "desc": "Coding & AI focus.",
                "link": "https://www.berlinsbi.com/programmes/undergraduate/bsc-hons-computer-science-and-digitisation"
            },
            "data_analytics": {
                "name": "MSc Data Analytics",
                "duration": "18 months",
                "campus": "Berlin, Hamburg",
                "desc": "Big data & Machine Learning.",
                "link": "https://www.berlinsbi.com/programmes/postgraduate/msc-data-analytics"
            },
            "digital_marketing_msc": {
                "name": "MSc Digital Marketing",
                "duration": "18 months",
                "campus": "Berlin, Hamburg, Paris, Barcelona",
                "desc": "Analytics & CRM.",
                "link": "https://www.berlinsbi.com/programmes/postgraduate/msc-digital-marketing"
            },
            "dba": {
                "name": "Doctorate in Business Administration (DBA)",
                "duration": "3-4 years",
                "campus": "Berlin",
                "desc": "Research & leadership.",
                "link": "https://www.berlinsbi.com/programmes/doctorate/dba-doctorate-in-business-administration"
            },
            "ba_comic_and_concept_art": {
                "name": "BA (Hons) Comic and Concept Art",
                "duration": "3 years",
                "campus": "Hamburg",
                "desc": "Focused on storytelling, illustration and concept art for games and media.",
                "link": "https://www.berlinsbi.com/programmes/undergraduate/ba-hons-comic-and-concept-art"
            },
            "ba_game_design": {
                "name": "BA (Hons) Game Design",
                "duration": "3 years",
                "campus": "Hamburg",
                "desc": "Game design theory and practice — level design, mechanics and production.",
                "link": "https://www.berlinsbi.com/programmes/undergraduate/ba-hons-game-design"
            },
            "ba_media_and_communications": {
                "name": "BA (Hons) Media and Communications",
                "duration": "3 years",
                "campus": "Hamburg",
                "desc": "Covers media theory, production and communication strategies.",
                "link": "https://www.berlinsbi.com/programmes/undergraduate/ba-hons-media-and-communications"
            },
            "ba_photography": {
                "name": "BA (Hons) Photography",
                "duration": "3 years",
                "campus": "Hamburg",
                "desc": "Practice-led photography programme with portfolio development.",
                "link": "https://www.berlinsbi.com/programmes/undergraduate/ba-hons-photography"
            },
            "bsc_international_business_and_management": {
                "name": "BSc (Hons) International Business and Management",
                "duration": "3 years",
                "campus": "Berlin",
                "desc": "Global business foundations, management and international trade.",
                "link": "https://www.berlinsbi.com/programmes/undergraduate/bsc-hons-international-business-and-management"
            },
            "msc_artificial_intelligence": {
                "name": "MSc Artificial Intelligence",
                "duration": "18 months",
                "campus": "Berlin, Hamburg",
                "desc": "Advanced AI topics: machine learning, NLP and applied AI systems.",
                "link": "https://www.berlinsbi.com/programmes/postgraduate/msc-artificial-intelligence"
            },
            "msc_health_psychology": {
                "name": "MSc Health Psychology",
                "duration": "18 months",
                "campus": "Berlin",
                "desc": "Psychology applied to health contexts, wellbeing and interventions.",
                "link": "https://www.berlinsbi.com/programmes/postgraduate/msc-health-psychology"
            },
            "msc_engineering_management": {
                "name": "MSc in Engineering Management",
                "duration": "18 months",
                "campus": "Berlin",
                "desc": "Combines engineering principles with management and leadership skills.",
                "link": "https://www.berlinsbi.com/programmes/postgraduate/msc-in-engineering-management"
            },
            "executive_mba": {
                "name": "Executive MBA",
                "duration": "18-24 months",
                "campus": "Berlin",
                "desc": "Leadership and strategic management for experienced professionals.",
                "link": "https://www.berlinsbi.com/programmes/postgraduate/executive-mba"
            },
            "ma_energy_management": {
                "name": "MA in Energy Management",
                "duration": "18 months",
                "campus": "Berlin",
                "desc": "Policy, business and technical aspects of the energy sector.",
                "link": "https://www.berlinsbi.com/programmes/postgraduate/ma-in-energy-management"
            },
            "ma_innovation_and_entrepreneurship": {
                "name": "MA Innovation & Entrepreneurship",
                "duration": "18 months",
                "campus": "Berlin",
                "desc": "Start-up creation, innovation methods and venture strategy.",
                "link": "https://www.berlinsbi.com/programmes/postgraduate/ma-in-innovation-and-entrepreneurship"
            },
            "ma_game_design": {
                "name": "MA Game Design",
                "duration": "18 months",
                "campus": "Hamburg",
                "desc": "Advanced game design, production and creative practice.",
                "link": "https://www.berlinsbi.com/programmes/postgraduate/ma-game-design"
            },
            "ma_user_experience_design": {
                "name": "MA User Experience Design",
                "duration": "18 months",
                "campus": "Berlin",
                "desc": "UX research, interaction design and user-centred product development.",
                "link": "https://www.berlinsbi.com/programmes/postgraduate/ma-user-experience-design"
            },
            "certificate_conversational_ai": {
                "name": "Conversational AI and Chatbot Systems (Certificate)",
                "duration": "Short course",
                "campus": "Online / Berlin",
                "desc": "A comprehensive introduction to conversational AI and chatbots.",
                "link": "https://www.berlinsbi.com/programmes/certificate-programmes/conversational-ai-and-chatbot-systems"
            },
            "certificate_iot_for_business_management": {
                "name": "IoT for Business Management (Certificate)",
                "duration": "Short course",
                "campus": "Berlin",
                "desc": "IoT concepts and applications for business management.",
                "link": "https://www.berlinsbi.com/programmes/certificate-programmes/iot-for-business-management"
            },
            "certificate_corporate_sustainability_leadership": {
                "name": "Corporate Sustainability Leadership (Certificate)",
                "duration": "Short course",
                "campus": "Berlin",
                "desc": "Sustainability frameworks and corporate strategies.",
                "link": "https://www.berlinsbi.com/programmes/certificate-programmes/corporate-sustainability-leadership"
            }
        }

        # Simple search logic: Check if any key is in the user's slot
        found_program = None
        
        if program_name in knowledge_base:
            found_program = knowledge_base[program_name]
        
        if not found_program:
            program_clean = program_name.lower()
            for key, details in knowledge_base.items():
                key = key.replace('_', ' ').replace("&", "and")
                print(f"Finding {program_clean} in {key} or {details['name']}")
                if key in program_clean or key in details['name']:
                    found_program = details
                    break
        
        if found_program:
            response = (f"🎓 **{found_program['name']}**\n"
                        f"⏳ Duration: {found_program['duration']}\n"
                        f"📍 Campus: {found_program['campus']}\n"
                        f"ℹ️ {found_program['desc']}\n"
                        f"🔗 Apply here: {found_program['link']}")
            dispatcher.utter_message(text=response)
        else:
            fallback_msg = f"I see you're interested in '{program_name}', but I couldn't find the specific details. "
            fallback_msg += "Try asking for 'MBA', 'Psychology', 'Animation', 'AI', 'Finance', or 'Tourism'."
            dispatcher.utter_message(text=fallback_msg)
            response = fallback_msg

        print('+'*50)
        print(f"Found program: {found_program}")
        print(f"Response: {response}")
        print('+'*50)
        return []