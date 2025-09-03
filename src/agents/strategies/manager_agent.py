# src/agents/manager_agent.py
from src.agents.base_agent import Analyst

class ManagerAgent(Analyst):
    """Manager analyst with access to all tools including LLM-based knowledge generation"""

    def __init__(self):
        super().__init__(
            name="Manager",
            description="A manager analyst with access to all tools including LLM-based knowledge generation.",
            tools=["vqa_tool", "wikipedia", "llm_knowledge"],
            system_prompt = """
            You are an AI assistant executing a task. Analyze the current state of your progress and decide the next best action.
            
            ## Available Tools:
                - **vqa_tool**: Use this tool for Visual Question Answering directly on the image to get initial candidate answers.
                - **wikipedia**: Use this tool to retrieve factual, encyclopedic knowledge from external sources relevant to the question.
                - **llm_knowledge**: Generate background knowledge and common-sense context using the LLM.
            
            ## Task
            Your goal is gather all the information to answer the user's question based on the provided image and context.
            To build a complete answer, you should use all available tools to gather all types of Information. 
            User Question: {question}
            Context: {context}

            ## Current Progress Summary
            - Tool Calls Made: {count_of_tool_calls}

            ## Information Gathered 
            ### Answer Candidate:
            {answer_candidate}

            ### LLM_Knowledge:
            {LLM_Knowledge}

            ### Factual Knowledge:
            {kbs_knowledge}

            ## Your Decision
            Based on the information you have, carefully review the user's question again.
            - If you have enough information to provide a complete and accurate answer, your next action is return "Finish".
            - If the current information is insufficient, choose ONE tool from the available list to gather the missing information. Do not repeat a tool call if you already have the necessary information.
            """,


            rationale_system_prompt="""
                Your task is to generate a logical explanation in Vietnamese. Do not include a final concluding sentence. Synthesize the visual details from 'Candidates', 'Context', 'Object_Analysis', with the facts from 'KBs_Knowledge'.
                Important: The 'Candidates' list is a suggestion and may be misleading or entirely incorrect.

                ### EXAMPLE 1
                Context: A wooden dining table is shown with a glossy finish.
                Question: Bàn được làm bằng gì?
                Candidates: Gỗ (0.92), Kim loại (0.05), Nhựa (0.02), Đá (0.01), Kính (0.00)
                KBs_Knowledge: Materials that are brown, smooth, and shiny are often polished or varnished wood.
                LLM_Knowledge: A dining table is a piece of furniture for eating meals. Wood is a common material for tables and can be given a glossy finish with varnish or polish.
                Rationale: Bề mặt của bàn có màu nâu, mịn và sáng bóng, là đặc điểm của gỗ.

                ### EXAMPLE 2
                Context: A photo of a single banana that has been partially peeled.
                Question: Quả chuối có đóng không?
                Candidates: không (0.95), có (0.05), bị thối (0.00), còn xanh (0.00), bằng nhựa (0.00)
                KBs_Knowledge: A banana is considered 'not closed' or 'open' when its peel is removed to expose the edible fruit inside.
                LLM_Knowledge: Peeling a fruit involves removing its outer skin. An 'open' or 'unclosed' state implies that this process has started, revealing the interior.
                Rationale: Quả chuối đã được bóc vỏ một phần, làm lộ ra phần ruột bên trong.

                ### EXAMPLE 3
                Context: A herd of zebras are gathered on a grassy field.
                Question: Các con vật đang làm gì?
                Candidates: Gặm cỏ (0.88), Đứng im (0.09), Chạy (0.02), Uống nước (0.01), Nằm nghỉ (0.00)
                KBs_Knowledge: Herbivores like zebras eat grass by lowering their mouths to the ground. This action is called grazing.
                LLM_Knowledge: Zebras are herbivores whose diet consists of grasses. Grazing is the act of feeding on grass, which involves an animal lowering its head to the ground.
                Rationale: Những con ngựa vằn đang cúi đầu và miệng của chúng ở gần bãi cỏ.

                ### EXAMPLE 4 
                Context: Two police officers on horseback patrolling a city street.
                Question: Người đàn ông đang làm gì?
                Candidates: đi bộ (0.65), đứng im (0.20), nói chuyện (0.10), chạy (0.05)
                KBs_Knowledge: The action of sitting on and controlling a horse is called 'riding a horse' (cưỡi ngựa).
                LLM_Knowledge: Police officers on horseback are known as mounted police. The action of controlling a horse while seated on its back is called 'riding'.
                Rationale: Phân tích hình ảnh cho thấy hai người đàn ông đang ngồi trên lưng ngựa.
                ### END OF EXAMPLES

                ### Now solve the new task
                Context: {context}
                Question: {question}
                Candidates: {candidates}
                KBs_Knowledge: {KBs_Knowledge}
                LLM_Knowledge: {LLM_Knowledge}
                Rationale:
            """,
            final_system_prompt="""
                You are an visual-question-answering assistant that generates the most accurate answer based on evidence.
                For each task you receive:
                - **Context:** <plain-text description of the image or scene>
                - **Question:** <single question>
                - **Candidates:** <A list of possible answers with probabilities, generated by a vision model>
                - **Rationale:** <The core reasoning that justifies the final answer>

                ### Instructions
                1. Read **Context**, **Question**, **Candidates**, and **Rationale** carefully.
                2. The 'Candidates' are only suggestions and may be incorrect. Your final answer must be based on the evidence in the `Rationale`. If the `Rationale` points to an answer not in the `Candidates` list, you must ignore the list.
                3. Answer each question concisely in a single word or short phrase.
                4. The final answer MUST be in Vietnamese, matching the language of the Question.

                ### EXAMPLE 1
                Context: A wooden dining table is shown with a glossy finish.
                Question: Bàn được làm bằng gì?
                Candidates: Gỗ (0.92), Kim loại (0.05), Nhựa (0.02), Đá (0.01), Kính (0.00)
                Rationale: Bề mặt của bàn có màu nâu, mịn và sáng bóng, là đặc điểm của gỗ.
                Answer: Gỗ

                ### EXAMPLE 2
                Context: A photo of a single banana that has been partially peeled.
                Question: Quả chuối có đóng không?
                Candidates: không (0.95), có (0.05), bị thối (0.00), còn xanh (0.00), bằng nhựa (0.00)
                Rationale: Quả chuối đã được bóc vỏ một phần, làm lộ ra phần ruột bên trong.
                Answer: không

                ### EXAMPLE 3
                Context: A herd of zebras are gathered on a grassy field.
                Question: Các con vật đang làm gì?
                Candidates: Gặm cỏ (0.88), Đứng im (0.09), Chạy (0.02), Uống nước (0.01), Nằm nghỉ (0.00)
                Rationale: Những con ngựa vằn đang cúi đầu và miệng của chúng ở gần bãi cỏ.
                Answer: Gặm cỏ

                ### EXAMPLE 4 
                Context: Two police officers on horseback patrolling a city street.
                Question: Người đàn ông đang làm gì?
                Candidates: đi bộ (0.65), đứng im (0.20), nói chuyện (0.10), chạy (0.05)
                Rationale: Phân tích hình ảnh cho thấy hai người đàn ông đang ngồi trên lưng ngựa, một hành động được gọi là 'cưỡi ngựa'. Danh sách ứng viên không chứa đáp án đúng này.
                Answer: cưỡi ngựa
                ### END OF EXAMPLE

                ### Now solve the new task
                Context: {context}
                Question: {question}
                Candidates: {candidates}
                Rationale: {rationale}
                Answer:
            """
        )

def create_manager_agent() -> ManagerAgent:
    """Factory function to create manager agent"""
    return ManagerAgent()