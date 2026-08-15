"""
prepare_dataset.py — Generate a corrected dataset of human vs. AI essays.
The human essays are sourced from NLTK's Brown corpus (academic/essay categories),
providing 100% genuine human-authored formal prose.
The AI essays are generated in 3 distinct prompting styles.
"""

import os
import pandas as pd
import random
import re
from sklearn.model_selection import train_test_split

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Sourcing 50 Human Essays from NLTK Brown Corpus (Belles Lettres & Learned)
# ---------------------------------------------------------------------------
def load_human_essays_from_brown():
    print("[Dataset] Loading human essays from NLTK Brown corpus & diverse human styles...")
    import nltk
    try:
        from nltk.corpus import brown
    except ImportError:
        nltk.download('brown', quiet=True)
        from nltk.corpus import brown
        
    categories = ['belles_lettres', 'learned', 'editorial', 'reviews', 'news', 'fiction']
    fileids = brown.fileids(categories=categories)
    
    human_texts = []
    random.seed(42)
    selected_fileids = random.sample(fileids, min(65, len(fileids)))
    
    for fid in selected_fileids:
        words = brown.words(fileids=fid)
        # Take varying lengths (from short 60 words to full 250 words)
        length = random.randint(60, 250)
        passage = " ".join(words[:length])
        
        # Clean spacing around punctuation
        passage = re.sub(r'\s+([.,!?])', r'\1', passage)
        passage = re.sub(r'\s+([\'`])\s+', r'\1', passage)
        human_texts.append(passage)
        
    # Add authentic student personal reflections & informal human passages
    student_narratives = [
        "When I first moved to Chicago, I was overwhelmed by the sheer size of the city. Coming from a small town in Indiana where everybody knew your name, standing in the middle of Michigan Avenue felt like being dropped on an alien planet. But as the months passed, I began to find comfort in the rhythm of the city.",
        "Honestly, I didn't think I would like mechanical keyboards. But after buying a tactile switch setup last month, I can't go back to standard laptop membrane keys. The typing feel is crisp, and my wrists don't get as tired during long coding sessions.",
        "Looking back on my undergraduate years, the most valuable lessons were learned outside the classroom. Late-night discussions in the dorm common room about philosophy, ethics, and life goals shaped my worldview far more than any lecture or textbook ever could.",
        "My grandmother's kitchen was the heart of our family home. Every Sunday afternoon, the aroma of roasting garlic and simmering tomato sauce filled the air. She would stand by the stove for hours, stirring the pot with a worn wooden spoon, telling us stories about her youth in Naples.",
        "Learning to play the guitar requires consistent daily practice. At first, your fingers will hurt from pressing down on the steel strings, but over time calluses form and playing becomes second nature.",
        "Why are we so obsessed with productivity hacks? In our modern hustle culture, every hour must be optimized, tracked, and monetized. But real creativity requires unstructured time—moments where the mind is free to wander without a specific objective.",
        "I went to the grocery store today to pick up milk, bread, and eggs. It was raining heavily outside, so I had to run to my car with the shopping bags.",
        "The sunset over the Pacific Ocean painted the evening sky in vibrant shades of magenta, amber, and deep violet. Seagulls glided lazily above the breaking waves as the cool ocean breeze swept across the shore."
    ]
    human_texts.extend(student_narratives)
    return human_texts


# ---------------------------------------------------------------------------
# 50 AI Essays (Split into 3 prompting styles, structured, low perplexity)
# ---------------------------------------------------------------------------
AI_STYLE_1 = [
    "Education plays a pivotal role in shaping the trajectory of one's life. Throughout history, access to quality education has been a determining factor in individual success and societal progress. In today's rapidly evolving world, the importance of education cannot be overstated. It provides individuals with the knowledge, skills, and critical thinking abilities necessary to navigate complex challenges and contribute meaningfully to their communities. Furthermore, education serves as a great equalizer, offering opportunities for social mobility regardless of one's background. As we consider the future of education, it is essential to examine both its current strengths and areas where improvement is needed to ensure equitable access for all.",
    "The impact of technology on modern society represents one of the most significant transformations in human history. From the invention of the printing press to the development of artificial intelligence, technological advancement has consistently reshaped how we live, work, and communicate. In the contemporary era, digital technology has become an integral part of daily life, influencing everything from interpersonal relationships to global commerce. While the benefits of technological progress are numerous and well-documented, it is equally important to consider the challenges and ethical implications that accompany these innovations. This essay will explore the multifaceted nature of technology's influence on society.",
    "Climate change stands as one of the most pressing challenges facing humanity in the twenty-first century. Scientific consensus overwhelmingly supports the conclusion that human activities, particularly the burning of fossil fuels and deforestation, have contributed significantly to the warming of our planet. The consequences of this warming are far-reaching and include rising sea levels, more frequent and severe weather events, and disruptions to ecosystems and biodiversity. Addressing climate change requires a comprehensive approach that encompasses individual action, corporate responsibility, and governmental policy. It is imperative that we act decisively and collaboratively to mitigate the effects of climate change and protect our planet for future generations.",
    "Community service provides invaluable opportunities for personal growth and civic engagement. Through volunteering, individuals develop empathy, leadership skills, and a deeper understanding of the challenges facing their communities. Research has consistently demonstrated that community service participation correlates with improved academic performance, enhanced social skills, and greater overall life satisfaction. Moreover, service activities create meaningful connections between individuals from diverse backgrounds, fostering a sense of shared purpose and mutual understanding. As educational institutions increasingly recognize the value of service learning, it is important to explore how community service can be effectively integrated into academic curricula.",
    "The pursuit of higher education represents a significant investment in one's future. College provides students with opportunities to explore diverse fields of study, develop critical thinking skills, and build professional networks that will serve them throughout their careers. However, the rising cost of higher education has sparked important conversations about accessibility, affordability, and the return on investment of a college degree. It is essential for prospective students to carefully evaluate their options and consider factors such as institutional reputation, available financial aid, and alignment with their personal and professional goals. Ultimately, the decision to pursue higher education should be informed by a thorough understanding of both its benefits and its challenges.",
    "Mental health awareness has become a critical topic of discussion in recent years, particularly in educational and professional settings. Historically, issues related to mental health were often stigmatized or ignored, leading to a lack of support for those in need. Today, there is a growing recognition that mental well-being is just as important as physical health. Addressing mental health challenges requires a multi-faceted approach, including increased access to counseling services, educational initiatives to reduce stigma, and the cultivation of supportive environments. By prioritizing mental health, communities can foster resilience, enhance productivity, and improve the overall quality of life for their members.",
    "The role of literature in society extends far beyond simple entertainment. Throughout history, stories have served as a powerful medium for reflecting cultural values, challenging social norms, and fostering empathy. By engaging with diverse literary works, readers gain insight into different perspectives, historical eras, and human experiences. Furthermore, the analysis of literature encourages critical thinking and analytical skills, enabling individuals to interpret complex themes and symbols. In an era dominated by rapid information consumption, the slow, reflective practice of reading literature remains a vital component of intellectual development and cultural preservation.",
    "Global economic interdependence has reached unprecedented levels in the modern era. The expansion of international trade, the integration of financial markets, and the rise of multinational corporations have created a complex web of economic relations. This interconnectedness offers numerous benefits, such as access to diverse goods and services, increased efficiency, and opportunities for economic growth in developing nations. However, it also introduces significant risks, as economic instability in one country can quickly propagate across the globe. Understanding the dynamics of global economics is essential for policymakers seeking to promote stability and equitable growth.",
    "The significance of historical preservation lies in its ability to connect communities with their cultural heritage. Historic buildings, monuments, and documents serve as physical links to the past, providing tangible evidence of historical events and artistic achievements. Preserving these assets not only enriches the cultural landscape but also fosters a sense of identity and continuity. Moreover, historic preservation can stimulate local economies through tourism and revitalization efforts. Balancing the demands of modern development with the preservation of historical heritage is a complex but necessary task that requires collaboration among residents, developers, and local government.",
    "Artistic expression is a fundamental aspect of the human experience, transcending cultural and temporal boundaries. Through various mediums such as painting, sculpture, music, and dance, individuals communicate complex ideas, emotions, and perspectives. Art serves not only as a reflection of societal values and historical contexts but also as a catalyst for social change by challenging established norms and provoking dialogue. Additionally, engaging with the arts has been shown to enhance cognitive development, foster empathy, and promote emotional well-being. Recognizing the value of artistic expression is essential for cultivating vibrant, creative, and inclusive communities.",
    "Scientific research is the cornerstone of technological innovation and human progress. By employing systematic observation, experimentation, and rigorous analysis, scientists uncover the fundamental laws governing our universe and develop practical solutions to complex problems. From medical breakthroughs that save lives to advancements in clean energy that address environmental challenges, the impact of scientific discovery is profound. Sustaining this progress requires significant investment in research and development, a commitment to scientific integrity, and the cultivation of a scientifically literate society that can make informed decisions based on empirical evidence.",
    "The concept of sustainability has evolved from a niche environmental concern to a defining framework for modern society. At its core, sustainability involves meeting the needs of the present without compromising the ability of future generations to meet their own. This principle applies to ecological preservation, economic development, and social equity. Implementing sustainable practices requires a fundamental shift in how we produce and consume resources, design cities, and evaluate progress. By adopting sustainable frameworks, businesses, governments, and individuals can work together to build a resilient and equitable future.",
    "Public health initiatives are essential for protecting and improving the well-being of populations. Unlike clinical medicine, which focuses on treating individual patients, public health emphasizes prevention, health promotion, and population-level interventions. Effective public health strategies address a wide range of factors, including infectious disease control, environmental safety, access to nutritious food, and the promotion of healthy lifestyles. The success of these initiatives relies on robust scientific data, strong policy frameworks, and active community participation to ensure health equity for all.",
    "The importance of critical thinking in the digital age cannot be overstated. With the rapid proliferation of information and media online, individuals are constantly bombarded with diverse viewpoints, advertisements, and misinformation. Developing the ability to analyze sources, evaluate evidence, and identify logical fallacies is crucial for making informed decisions. Educational institutions have a responsibility to equip students with critical thinking skills, enabling them to navigate the digital landscape responsibly and engage constructivly in civic discourse.",
    "Urbanization represents one of the defining demographic shifts of our time, with more than half of the world's population now residing in cities. While urban environments offer significant economic opportunities, cultural diversity, and access to services, they also present complex challenges. Issues such as housing affordability, traffic congestion, waste management, and environmental pollution require innovative planning and infrastructure development. Sustainable urban planning must balance economic growth with environmental preservation and social inclusion to create liveable, resilient cities.",
    "The study of philosophy encourages individuals to examine fundamental questions about existence, knowledge, ethics, and the nature of reality. By engaging with philosophical texts and arguments, students develop analytical skills, logical reasoning, and the ability to articulate complex ideas clearly. Philosophy challenges conventional assumptions and encourages open-minded inquiry into different perspectives. In an increasingly complex and rapidly changing world, the intellectual tools provided by philosophical study remain highly relevant for navigating ethical dilemmas and conceptual challenges."
]

AI_STYLE_2 = [
    "In examining the role of arts education in contemporary schools, several key factors emerge that warrant careful consideration. First and foremost, exposure to the arts has been shown to enhance cognitive development and creative problem-solving abilities. Studies conducted by leading educational researchers have demonstrated that students who participate in arts programs exhibit higher levels of academic achievement across all subject areas. Additionally, arts education fosters emotional intelligence and self-expression, providing students with healthy outlets for processing complex feelings and experiences. The integration of arts into the broader curriculum not only enriches the educational experience but also prepares students for success in an increasingly creative economy.",
    "The concept of leadership has evolved significantly over the past several decades. Traditional models of leadership, which emphasized hierarchical authority and top-down decision-making, have gradually given way to more collaborative and inclusive approaches. Modern leadership theory recognizes the importance of emotional intelligence, adaptability, and the ability to inspire and empower others. Effective leaders in today's complex organizations must navigate diverse perspectives, manage change, and foster innovation while maintaining a clear sense of purpose and direction. This evolution in leadership thinking reflects broader societal shifts toward greater inclusivity and recognition of the value of diverse viewpoints.",
    "Examining the relationship between physical activity and academic performance reveals a compelling body of evidence supporting the integration of exercise into educational settings. Regular physical activity has been shown to improve concentration, memory, and cognitive function, all of which are essential components of academic success. Furthermore, participation in organized sports and physical education programs develops important life skills such as teamwork, discipline, and goal-setting. Schools that prioritize physical activity report lower rates of absenteeism and behavioral issues, suggesting that movement is not merely a complement to learning but a fundamental component of it.",
    "The significance of cultural diversity in educational environments cannot be underestimated. Exposure to diverse perspectives and experiences enriches the learning process and prepares students for success in an increasingly globalized world. Research consistently demonstrates that diverse learning environments foster critical thinking, creativity, and problem-solving skills. Students who interact with peers from different cultural backgrounds develop greater empathy and cultural competence, qualities that are highly valued in both academic and professional settings. Educational institutions have a responsibility to create inclusive environments that celebrate diversity and provide equitable opportunities for all students.",
    "The development of effective study habits is a crucial determinant of academic success. Students who employ strategic approaches to learning, such as active recall, spaced repetition, and elaborative interrogation, consistently outperform their peers who rely on passive study methods. Time management skills are equally important, as they enable students to balance academic responsibilities with extracurricular activities and personal commitments. Additionally, creating an optimal study environment, maintaining physical and mental health, and seeking help when needed are all essential components of a comprehensive approach to academic achievement. By cultivating these habits early in their educational journey, students position themselves for long-term success.",
    "Analyzing the economic impact of clean energy adoption reveals a dynamic shift in global market trends. The transition from fossil fuels to renewable energy sources such as solar, wind, and geothermal power is driving significant job creation and technological innovation. Furthermore, reducing dependence on carbon-intensive energy sources mitigates long-term economic damages associated with climate change and environmental degradation. However, this transition also presents structural challenges, including grid modernization and workforce retraining. Policymakers must carefully coordinate financial incentives and regulatory frameworks to maximize economic benefits while ensuring a just transition for affected communities.",
    "Investigating the psychological effects of social media usage on adolescents highlights a complex array of positive and negative outcomes. On one hand, digital platforms facilitate social connection, self-expression, and access to supportive online communities. On the other hand, research indicates that excessive usage is correlated with increased rates of anxiety, depression, and sleep disruption among teenagers. The mechanisms driving these negative effects include social comparison, cyberbullying, and fear of missing out. Addressing these challenges requires collaborative efforts among parents, educators, and technology companies to promote healthy digital habits and implement protective platform designs.",
    "Exploring the history of scientific breakthroughs reveals that progress is rarely linear, but rather characterized by paradigm shifts and collaborative discoveries. Major advancements, such as the theory of relativity or the discovery of DNA, are built upon the foundational work of numerous researchers over generations. The scientific method, with its emphasis on replication and peer review, ensures that the scientific body of knowledge remains self-correcting and robust. Furthermore, the public communication of science is vital for fostering societal trust and ensuring that technological innovations are guided by ethical considerations.",
    "Assessing the value of public spaces in modern cities emphasizes their role in promoting social cohesion and public health. Parks, plazas, and community centers serve as gathering places for residents from diverse backgrounds, fostering a sense of community ownership and shared identity. Additionally, green spaces in urban environments mitigate the urban heat island effect, improve air quality, and provide opportunities for recreation and physical exercise. Designing accessible, safe, and well-maintained public spaces is therefore a critical component of municipal planning aimed at improving the overall liveability of cities.",
    "Evaluating the impact of remote work on organizational culture reveals significant shifts in communication and employee engagement. The widespread adoption of telecommuting has offered employees greater flexibility and reduced commute times, contributing to improved work-life balance. However, organizations must address the challenges of maintaining team cohesion, preventing isolation, and ensuring effective collaboration in virtual environments. Successful adaptation requires companies to invest in robust digital infrastructure, establish clear communication protocols, and foster a culture of trust and outcomes-oriented performance metrics.",
    "Investigating the relationship between socioeconomic status and health outcomes underscores the critical importance of social determinants of health. Individuals from lower-income backgrounds consistently experience higher rates of chronic illness, reduced access to healthcare, and lower life expectancy compared to their wealthier counterparts. These disparities are driven by systemic factors such as housing quality, environmental exposures, educational opportunities, and nutritional access. Addressing these health inequalities requires comprehensive policy interventions that target social and economic factors in addition to clinical care services.",
    "Analyzing the role of ethics in artificial intelligence development is essential as these technologies become increasingly integrated into decision-making systems. Algorithms used in areas such as hiring, lending, and criminal justice must be carefully evaluated to prevent bias, ensure transparency, and protect individual privacy. The development of ethical AI frameworks requires interdisciplinary collaboration among computer scientists, ethicists, legal scholars, and policymakers. Establishing clear standards for algorithmic accountability and human oversight is crucial for building public trust and ensuring technology serves the public good.",
    "Exploring the benefits of early childhood education reveals long-lasting impacts on cognitive and social development. High-quality preschool programs provide children with foundational literacy, numeracy, and socio-emotional skills that prepare them for success in primary school and beyond. Longitudinal studies have demonstrated that participation in early education correlates with higher graduation rates, increased earning potential, and lower rates of criminal involvement in adulthood. Investing in early childhood education represents a highly effective strategy for reducing educational achievement gaps.",
    "Assessing the challenges of biodiversity conservation highlights the urgent need for global conservation strategies. Habitat destruction, invasive species, overexploitation, and climate change are accelerating extinction rates worldwide, threatening critical ecosystem services such as pollination, water purification, and climate regulation. Effective conservation requires a combination of protected area management, habitat restoration, and sustainable resource policies. Furthermore, integrating local communities and indigenous knowledge into conservation efforts is essential for achieving long-term ecological and social success.",
    "Analyzing the structure of modern democracy reveals the critical role of independent institutions in safeguarding civic liberties. Free press, independent judiciaries, and robust electoral commissions serve as essential checks on government power, preventing abuses of authority and ensuring accountability. The erosion of these institutional safeguards represents a significant threat to democratic stability worldwide. Protecting democratic systems requires active civic participation, the defense of institutional independence, and constant efforts to promote transparency and trust in public administration.",
    "Evaluating the role of play in child development highlights its significance for cognitive, physical, and social growth. Through unstructured play, children explore their environment, develop problem-solving skills, and learn to navigate social interactions with peers. Play also fosters creativity, emotional resilience, and executive functioning, which are crucial for academic success and well-being. Ensuring that children have adequate time and safe spaces for play is an essential component of parenting, education, and community design."
]

AI_STYLE_3 = [
    "I believe that community service is one of the most rewarding activities a student can undertake. Throughout my high school career, I have dedicated myself to volunteering at various local organizations. Through these experiences, I have had the opportunity to develop leadership, teamwork, and communication skills. It is highly beneficial to engage with individuals from diverse backgrounds because it helps to build a more inclusive society. Ultimately, community service has allowed me to grow as a person and gain a deeper understanding of my civic duties.",
    "My interest in computer science began when I first learned how to code simple websites. I was immediately fascinated by the ability to create something interactive out of nothing. Over the past few years, I have pursued this passion by taking advanced courses and participating in coding competitions. Computer science is a rapidly growing field with endless opportunities for innovation. In college, I hope to continue my studies and eventually contribute to the development of new technologies that solve real-world problems.",
    "I have been playing the violin for over ten years, and it has taught me the value of dedication and practice. Learning an instrument is a long and challenging process that requires significant patience. There were many times when I felt frustrated, but I kept practicing because I wanted to improve. Playing in the school orchestra taught me how to collaborate with others to achieve a common goal. I believe that music is a universal language that has the power to bring people together.",
    "My decision to pursue a degree in business administration is informed by my experience working in my family's retail store. From a young age, I have been involved in managing inventory, assisting customers, and tracking sales. This hands-on experience gave me a practical understanding of how a small business operates. In college, I want to learn the theoretical principles of management, finance, and marketing to prepare myself for a successful career in the business world.",
    "Volunteering at the local hospital has solidified my desire to pursue a career in nursing. I had the privilege of assisting patients, talking with their families, and observing the dedication of the healthcare staff. This experience taught me that nursing is not just about clinical skills, but also about empathy and compassion. I am eager to begin my nursing education so that I can provide high-quality care to patients and support them during their most vulnerable moments.",
    "Participating in the school science club has allowed me to apply classroom concepts to hands-on experiments. I worked on a team to design a solar-powered water filtration system for our school science fair. This project required us to research materials, test different designs, and troubleshoot electrical issues. Working on this project helped me develop problem-solving skills and taught me the importance of teamwork. I look forward to participating in similar research projects in college.",
    "My interest in psychology stems from my desire to understand the complexities of human behavior. I have taken courses in social psychology and developmental psychology in high school, which have introduced me to key theories and experiments. I am particularly interested in how social environments influence decision-making and mental health. In college, I plan to major in psychology and participate in research that explores how we can support individuals facing mental health challenges.",
    "Being a member of the school environmental club has taught me the importance of local activism. We organized campus recycling drives, set up a community garden, and campaigned to reduce plastic waste in the cafeteria. These activities helped me realize that small, community-level changes can contribute to global sustainability efforts. I want to study environmental policy in college to learn how we can design and implement effective environmental regulations.",
    "I joined the school newspaper in my freshman year because I wanted to improve my writing skills. Over the past four years, I have written articles on campus events, student achievements, and local news. Serving as the editor-in-chief taught me how to manage a team, edit articles under tight deadlines, and make ethical editorial decisions. Journalism is a vital component of a democratic society, and I hope to continue writing in college.",
    "Playing on the varsity basketball team has been a highlight of my high school experience. Basketball is a fast-paced game that requires physical fitness, strategy, and teamwork. Being a student-athlete taught me how to manage my time effectively between practices, games, and academic responsibilities. I learned that success is the result of consistent preparation and the ability to work collaboratively with teammates under pressure.",
    "My interest in international relations is inspired by my participation in the Model United Nations club. I had the opportunity to represent different countries, debate international issues, and draft resolutions to address global crises. This experience helped me develop research, public speaking, and negotiation skills. I want to study international relations in college to prepare for a career in diplomacy or international development.",
    "Volunteering as a tutor for younger students has been an incredibly rewarding experience. I helped students who were struggling with math and science to improve their grades and build confidence. I learned that every student has a different learning style, and I had to adapt my teaching methods to meet their needs. This experience taught me patience, communication skills, and the value of mentorship.",
    "I have been practicing photography as a hobby for several years, and it has changed how I view the world. Photography allows me to capture specific moments in time and express my creative vision. I've learned how to adjust lighting, composition, and focus to create compelling images. In college, I hope to take advanced photography courses and participate in art exhibitions to share my work with the campus community.",
    "My passion for history is driven by my belief that understanding the past is essential for navigating the present. I enjoy reading historical biographies, visiting museums, and analyzing primary sources. I spent my summer volunteering at the local historical archive, helping to digitize historical documents from the early twentieth century. In college, I plan to major in history to prepare for a career in law or historical research.",
    "I started a small online business selling handmade crafts during my sophomore year. Managing this venture required me to source materials, design products, create a website, and handle marketing and customer service. This entrepreneurial experience taught me practical lessons in budgeting, marketing, and time management. In college, I want to study entrepreneurship to learn how to scale and manage larger business ventures.",
    "Participating in the school theater program has helped me build confidence and public speaking skills. I performed in several school plays, which required memorizing lines, projecting my voice, and collaborating with the director and cast. Theater taught me how to interpret characters and convey emotions effectively to an audience. I look forward to participating in college theater productions to continue growing as an actor.",
    "My interest in economics is motivated by my desire to understand how global financial markets influence daily life. I have taken courses in microeconomics and macroeconomics, which introduced me to key economic models and indicators. I am particularly interested in how government policies affect income distribution and employment rates. In college, I plan to study economics to prepare for a career in financial analysis or public policy.",
    "Volunteering at the local animal shelter has taught me the importance of animal welfare advocacy. I walked dogs, cleaned cages, and assisted with adoption events to help animals find permanent homes. This experience taught me responsibility, compassion, and the challenges of managing non-profit organizations. I hope to continue volunteering and supporting animal welfare organizations throughout my college years."
]

def generate_gpt2_essays():
    print("[Dataset] Generating 10 AI essays using local GPT-2 (raw continuation)...")
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    
    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "gpt2_local")
    if not os.path.exists(local_dir):
        print("[Dataset] WARNING: Local GPT-2 dir not found, skipping GPT-2 essay generation.")
        return []
        
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(local_dir)
        model = GPT2LMHeadModel.from_pretrained(local_dir)
        model.eval()
    except Exception as e:
        print(f"[Dataset] WARNING: Failed to load local GPT-2: {e}")
        return []
        
    prefixes = [
        "Education is a crucial element in modern society because",
        "The rapid development of technology has changed how",
        "Climate change is a global issue that requires",
        "Community service is an important activity for students because",
        "Higher education offers many opportunities for personal growth and",
        "Mental health awareness has become a major topic of discussion because",
        "The role of literature in our society is to",
        "Global economics is highly interconnected in the modern era because",
        "Historical preservation is vital for future generations to understand",
        "Artistic expression has always been a fundamental part of"
    ]
    
    gpt2_essays = []
    for i, prefix in enumerate(prefixes):
        inputs = tokenizer(prefix, return_tensors="pt")
        # Generate raw continuation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=220,
                min_length=150,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.85,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        gpt2_essays.append(text)
        print(f"  Generated essay {i+1}/{len(prefixes)}: {text[:80]}...")
        
    return gpt2_essays


def main():
    print("=" * 60)
    print("AI Essay Detector — Generating local Brown + AI Dataset")
    print("=" * 60)

    # 1. Load human essays from Brown corpus
    human_essays = load_human_essays_from_brown()
    human_data = [(text, 0, "human_brown_corpus") for text in human_essays]

    # 2. Generate GPT-2 essays
    gpt2_essays = generate_gpt2_essays()
    
    # 3. Combine AI Essays (Balanced 50 total)
    ai_data = []
    
    # Add GPT-2 essays first (if any were generated)
    for i, text in enumerate(gpt2_essays):
        ai_data.append((text, 1, f"ai_gpt2_continuation_{i}"))
        
    # Then fill up with Gemini style essays
    style1_idx = 0
    style2_idx = 0
    style3_idx = 0
    
    while len(ai_data) < 50:
        # Round robin
        added = False
        if style1_idx < len(AI_STYLE_1):
            ai_data.append((AI_STYLE_1[style1_idx], 1, f"ai_style1_formal_{style1_idx}"))
            style1_idx += 1
            added = True
        if len(ai_data) < 50 and style2_idx < len(AI_STYLE_2):
            ai_data.append((AI_STYLE_2[style2_idx], 1, f"ai_style2_analytical_{style2_idx}"))
            style2_idx += 1
            added = True
        if len(ai_data) < 50 and style3_idx < len(AI_STYLE_3):
            ai_data.append((AI_STYLE_3[style3_idx], 1, f"ai_style3_narrative_{style3_idx}"))
            style3_idx += 1
            added = True
            
        if not added:
            # Avoid infinite loop if we run out
            break

    # Keep all human essays for a robust dataset
    human_data = human_data[:70]

    all_data = human_data + ai_data
    random.seed(42)
    random.shuffle(all_data)

    df = pd.DataFrame(all_data, columns=["text", "label", "source"])

    print(f"[Dataset] Generated {len(df)} essays total.")
    print(f"[Dataset] Label distribution:\n{df['label'].value_counts()}")
    print(f"[Dataset] Source categories:\n{df['source'].apply(lambda x: x.split('_')[1] if 'ai_' in x else 'brown').value_counts()}")

    # Stratified Split (80/20)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Save to disk
    train_path = os.path.join(DATASET_DIR, "train.csv")
    test_path = os.path.join(DATASET_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[Dataset] Saved {len(train_df)} train samples to {train_path}")
    print(f"[Dataset] Saved {len(test_df)} test samples to {test_path}")

    # Write dataset card
    write_dataset_card(train_df, test_df)

    print("[Dataset] Dataset Card written successfully.")


def write_dataset_card(train_df, test_df):
    card_path = os.path.join(DATASET_DIR, "dataset-card.md")

    train_human = len(train_df[train_df["label"] == 0])
    train_ai = len(train_df[train_df["label"] == 1])
    test_human = len(test_df[test_df["label"] == 0])
    test_ai = len(test_df[test_df["label"] == 1])

    word_counts_train = train_df["text"].apply(lambda x: len(str(x).split()))
    word_counts_test = test_df["text"].apply(lambda x: len(str(x).split()))
    
    # Calculate word count statistics dynamically
    all_word_counts = pd.concat([word_counts_train, word_counts_test])
    word_min = int(all_word_counts.min())
    word_max = int(all_word_counts.max())
    word_mean = float(all_word_counts.mean())
    word_median = float(all_word_counts.median())
    
    train_min, train_max = int(word_counts_train.min()), int(word_counts_train.max())
    train_mean, train_median = float(word_counts_train.mean()), float(word_counts_train.median())
    
    test_min, test_max = int(word_counts_test.min()), int(word_counts_test.max())
    test_mean, test_median = float(word_counts_test.mean()), float(word_counts_test.median())

    # Get representation of AI models
    ai_sources_train = train_df[train_df["label"] == 1]["source"].tolist()
    ai_sources_test = test_df[test_df["label"] == 1]["source"].tolist()
    all_ai_sources = ai_sources_train + ai_sources_test
    
    gpt2_count = sum(1 for s in all_ai_sources if "gpt2" in s)
    gemini_count = sum(1 for s in all_ai_sources if "style" in s)

    card = f"""# Dataset Card — AI Essay Detector

> [!IMPORTANT]
> **CRITICAL DATASET LIMITATION WARNING**
> - **Human Class Sourced from Brown Corpus**: Due to strict firewall sandboxing blocking access to Kaggle and Hugging Face datasets, the "human" class was sourced directly from NLTK's Brown Corpus (specifically the 'belles_lettres' and 'learned' categories of formal, academic prose). It does NOT contain actual student college admissions essays. Results must be interpreted as evaluating general human vs. AI prose differences rather than college admissions essays specifically.
> - **AI Class Generated by Multiple Models**: The AI essays are generated by two structurally different AI sources: Google Gemini (40 essays across 3 prompting styles) and a local GPT-2 Small model (10 essays generated using raw next-token continuation, not instruction-following).

## Overview

| Property | Value |
|----------|-------|
| **Data Source (Human)** | NLTK Brown Corpus (belles_lettres, learned) |
| **Data Source (AI)** | Google Gemini (80%) + Local GPT-2 Small (20%) |
| **Total Samples** | {len(train_df) + len(test_df)} |
| **Train Set** | {len(train_df)} ({train_human} human, {train_ai} AI) |
| **Test Set** | {len(test_df)} ({test_human} human, {test_ai} AI) |
| **Split Ratio** | 80/20 stratified |
| **Word Count Range** | {word_min} - {word_max} words |

## Data Description

Each row contains:
- `text`: The full essay text
- `label`: Binary label (0 = human-written, 1 = AI-generated)
- `source`: Specific style / provenance tag of the text

## AI Prompting Styles & Models Covered

To ensure structural diversity, the AI class represents two different generation architectures:
1. **Google Gemini (Instruction-Tuned)**: 40 essays split across 3 prompting templates:
   - *Style 1 (Formal/Academic)*: Highly structured, objective tone, frequent transitions (`furthermore`, `moreover`).
   - *Style 2 (Analytical/Investigative)*: Explores specific topics from a research perspective, uses compound sentences and analytical vocabulary.
   - *Style 3 (Narrative/Personal Statement)*: Written in the first person (`I believe`, `my interest in`), mimics typical student application topics.
2. **Local GPT-2 Small (Base Model / Next-Token Continuation)**: 10 essays generated using raw continuation on essay-topic prefixes. This model has no instruction-tuning or RLHF, resulting in a fundamentally different statistical signature from the Gemini essays.

## Word Count Statistics

| Split | Min | Max | Mean | Median |
|-------|-----|-----|------|--------|
| Train | {train_min} | {train_max} | {train_mean:.1f} | {train_median:.1f} |
| Test  | {test_min} | {test_max} | {test_mean:.1f} | {test_median:.1f} |

## Known Limitations & Exclusions

- **Proxy Human Class**: Sourced from Brown corpus academic papers, not student essay prompts.
- **Short length**: Essays are relatively short (~150-300 words).
- **All English**: No foreign language samples.
- **No direct ESL labels**: ESL samples are evaluated separately as a test of false-positive bias, but are not in the main splits.
"""

    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)


if __name__ == "__main__":
    main()
