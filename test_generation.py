
from generation import generate_answer

question = "What are my technical interests?"

retrieved_chunks = [
    "I am a Computer Science student interested "
    "in Artificial Intelligence and Python programming.",

    "I enjoy learning about machine learning "
    "and developing software applications."
]

answer = generate_answer(question, retrieved_chunks)

print("\nDIGITAL CLONE ANSWER:")
print(answer)
