from src.states.blogstate import BlogState
from langchain_core.messages import HumanMessage,AIMessage
from src.states.blogstate import Blog

class BlogNode:
    """
    A class to represent blog nodes"""

    def __init__(self,llm):
        self.llm=llm

    def title_Creation(self,state:BlogState):
        """
        create the title of the blog
        """
        if "topic" in state and state["topic"]:
            prompt= """
                You are an expert blog content writer. Use Markdown formatting. Generate
                   a blog title for the {topic}. Return only the final blog title as plain text, no markdown heading, no reasoning, no <think> blocks.
                    """

            system_message=prompt.format(topic=state["topic"])
            response=self.llm.invoke(system_message)

            return {'blog':{'title':response.content}}

    def blog_content_generation(self,state:BlogState):
        """
        content generation for the blog
        """

        if "topic" in state and state["topic"]:
            prompt= """You are expert blog writer. Use Markdown formatting.
                Generate a detailed blog content with detailed breakdown for the {topic} . Write only the final blog article in markdown.
            Do not include reasoning, analysis, or <think> blocks.
            Topic: {topic}     """
            system_message=prompt.format(topic=state["topic"])
            response=self.llm.invoke(system_message)

            return {'blog':{'title':state['blog']['title'],'content':response.content}}


    def translate(self,state:BlogState):
        """
        Tranlate the content to a specified language"""

        translation_prompt="""
            Tranlsate the following content into {current_language}.
            - Maintain the original tone,style and formatting.
            - Adapt cultural language refence and idioms to be appropiate for the {current_language}.

            ORIGINAL CONTENT:
            {blog_content}
        """
        blog_content=state['blog']['content']
        message=[
            HumanMessage(translation_prompt.format(current_language=state["current_language"],blog_content=blog_content))
        ]

        translation_content=self.llm.with_structured_output(Blog).invoke(message)

        return {"blog":{"content":translation_content}}

    def route(self,state:BlogState):
        return {"current_language":state["current_language"]}    

    def route_decission(self,state:BlogState):
        """
        route the content of the respective tranlsation function"""
        if state["current_language"]=="hindi":
            return "hindi"
        elif state["current_language"]=="french":
            return "french"
        else:
            return state["current_language"]


    
            