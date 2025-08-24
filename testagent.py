# test_agent.py
from agent import query_agent

def run_test():
    test_q = "What is Big Data?"  # example question from your PDF
    answer = query_agent(test_q)

    print("Question:", test_q)
    print("Answer:", answer)

if __name__ == "__main__":
    run_test()