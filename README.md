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

