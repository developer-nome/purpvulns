# Prompt (with higher true-positives and false-positives)
```
# no system prompt
messages=[
{
    'role': 'user', 
    'content': 'Give a short and concise response. Analyze the following ' + code_language + ' code for security vulnerabilities using OWASP guidelines and report any issues found in a brief summary. \n\n ' + your_variable_with_codeblock_here
}
```

# Prompt (with lower true-positives and false-positives, plus JSON output)
```

system_prompt = """
    Respond with a valid JSON string only, without any additional text or explanations. The output should adhere to the following structure:
    ```
    {
        "security_vulnerabilities": [
            {
                "vulnerability": "<string>",
                "description": "<string>"
            }
        ]
    }
    ```
    Do not include any introductory phrases or sentences in your response. Only provide the JSON object.
    """
    stream = await client.chat.completions.create(
        messages=[
        {
            'role': 'system', 
            'content': system_prompt,
        },
        {
            'role': 'user', 
            'content': 'Your task is to analyze the following ' + code_language + ' code for security vulnerabilities using OWASP guidelines and report any issues found in a brief summary. Ensure that your answers are not false positives. Listing any results that is not clearly a serious code vulnerability will result in a lower score on your assignment. Here is the code: \n\n ' + your_variable_with_codeblock_here
        }
    ],
    model = os.getenv("LLM_MODEL"), 
    stream=True,
    temperature=0.4,
    response_format={
        "type": "json_object"
    }
    )
```

# Top Tested LLMs
| Model    | Accuracy |
| -------- | ------- |
| Phi-3.5 (3.8B-Mini-Instruct-Q6_K)  | 96% |
| GPT 4o Mini | 94% |
| GLM 4 (9B-Chat-Q5_0  | 94% |
| SuperNova-Medius (Q4_K_M) | 94% |
| Llama 3.1 (70B) | 92% |
| Wizard LM2 (7B-Q6_K) | 92% |
| Claud 3 Haiku | 90% |
| Claude 3.5 Sonnet | 88% |
| GPT 4o | 88% |
| Llama 3.1 (405B) | 85% |
| Mistral Nemo | 85% |
| Llama 3.2 (3B-Instruct-Q6_K) | 83% |
| GPT 4 Turbo| 83% |
| OpenCodeInterpreter DS (6.7B-Q6_K)| 83% |
| Qwen 2.5 Coder (7B-Instruct-Q6_K) | 81% |
| CodeGemma | 79% |
| Llama 3.1 (8B-Instruct-Q6_K) | 79% |


