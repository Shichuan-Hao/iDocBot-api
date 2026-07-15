```bash
---------------------------------------------------------------------------
APIStatusError                            Traceback (most recent call last)
Cell In[33], line 7
      3 # 初始化 DeepSeek 的 API 客户端
      4 client = OpenAI(api_key="sk-1a26f9f0887943f48e1bff80ebf5971d", base_url="https://api.deepseek.com")
      5 
      6 # 调用 DeepSeek 的 API，生成回答
----> 7 response = client.chat.completions.create(
      8     model = "deepseek-chat",
      9     messages = [
     10         {

File ~/miniconda3/envs/iDocBot-api/lib/python3.11/site-packages/openai/_utils/_utils.py:286, in required_args.<locals>.inner.<locals>.wrapper(*args, **kwargs)
    284             msg = f"Missing required argument: {quote(missing[0])}"
    285     raise TypeError(msg)
--> 286 return func(*args, **kwargs)

File ~/miniconda3/envs/iDocBot-api/lib/python3.11/site-packages/openai/resources/chat/completions/completions.py:1211, in Completions.create(self, messages, model, audio, frequency_penalty, function_call, functions, logit_bias, logprobs, max_completion_tokens, max_tokens, metadata, modalities, n, parallel_tool_calls, prediction, presence_penalty, prompt_cache_key, prompt_cache_retention, reasoning_effort, response_format, safety_identifier, seed, service_tier, stop, store, stream, stream_options, temperature, tool_choice, tools, top_logprobs, top_p, user, verbosity, web_search_options, extra_headers, extra_query, extra_body, timeout)
   1164 @required_args(["messages", "model"], ["messages", "model", "stream"])
   1165 def create(
   1166     self,
   (...)   1208     timeout: float | httpx.Timeout | None | NotGiven = not_given,
   1209 ) -> ChatCompletion | Stream[ChatCompletionChunk]:
   1210     validate_response_format(response_format)
-> 1211     return self._post(
   1212         "/chat/completions",
   1213         body=maybe_transform(
   1214             {
   1215                 "messages": messages,
   1216                 "model": model,
   1217                 "audio": audio,
   1218                 "frequency_penalty": frequency_penalty,
   1219                 "function_call": function_call,
   1220                 "functions": functions,
   1221                 "logit_bias": logit_bias,
   1222                 "logprobs": logprobs,
   1223                 "max_completion_tokens": max_completion_tokens,
   1224                 "max_tokens": max_tokens,
   1225                 "metadata": metadata,
   1226                 "modalities": modalities,
   1227                 "n": n,
   1228                 "parallel_tool_calls": parallel_tool_calls,
   1229                 "prediction": prediction,
   1230                 "presence_penalty": presence_penalty,
   1231                 "prompt_cache_key": prompt_cache_key,
   1232                 "prompt_cache_retention": prompt_cache_retention,
   1233                 "reasoning_effort": reasoning_effort,
   1234                 "response_format": response_format,
   1235                 "safety_identifier": safety_identifier,
   1236                 "seed": seed,
   1237                 "service_tier": service_tier,
   1238                 "stop": stop,
   1239                 "store": store,
   1240                 "stream": stream,
   1241                 "stream_options": stream_options,
   1242                 "temperature": temperature,
   1243                 "tool_choice": tool_choice,
   1244                 "tools": tools,
   1245                 "top_logprobs": top_logprobs,
   1246                 "top_p": top_p,
   1247                 "user": user,
   1248                 "verbosity": verbosity,
   1249                 "web_search_options": web_search_options,
   1250             },
   1251             completion_create_params.CompletionCreateParamsStreaming
   1252             if stream
   1253             else completion_create_params.CompletionCreateParamsNonStreaming,
   1254         ),
   1255         options=make_request_options(
   1256             extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
   1257         ),
   1258         cast_to=ChatCompletion,
   1259         stream=stream or False,
   1260         stream_cls=Stream[ChatCompletionChunk],
   1261     )

File ~/miniconda3/envs/iDocBot-api/lib/python3.11/site-packages/openai/_base_client.py:1297, in SyncAPIClient.post(self, path, cast_to, body, content, options, files, stream, stream_cls)
   1288     warnings.warn(
   1289         "Passing raw bytes as `body` is deprecated and will be removed in a future version. "
   1290         "Please pass raw bytes via the `content` parameter instead.",
   1291         DeprecationWarning,
   1292         stacklevel=2,
   1293     )
   1294 opts = FinalRequestOptions.construct(
   1295     method="post", url=path, json_data=body, content=content, files=to_httpx_files(files), **options
   1296 )
-> 1297 return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))

File ~/miniconda3/envs/iDocBot-api/lib/python3.11/site-packages/openai/_base_client.py:1070, in SyncAPIClient.request(self, cast_to, options, stream, stream_cls)
   1067             err.response.read()
   1069         log.debug("Re-raising status error")
-> 1070         raise self._make_status_error_from_response(err.response) from None
   1072     break
   1074 assert response is not None, "could not resolve response (should never happen)"

APIStatusError: Error code: 402 - {'error': {'message': 'Insufficient Balance', 'type': 'unknown_error', 'param': None, 'code': 'invalid_request_error'}}
```


```bash
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[54], line 2
      1 from langchain.chat_models import init_chat_model
----> 2 from langchain.output_parsers.boolean import BooleanOutputParser
      3 
      4 prompt_template = ChatPromptTemplate(
      5     [

ModuleNotFoundError: No module named 'langchain.output_parsers'
```