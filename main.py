from dotenv                import dotenv_values
from consulta_vector_first import PipelineVectorFirst

def main():
    config   = dotenv_values(dotenv_path=".env")
    pipeline = PipelineVectorFirst(
        openai_api_key = config.get("OPENAI_API_KEY"),
        neo4j_uri      = config.get("NEO4J_URI"),
        neo4j_user     = config.get("NEO4J_USER"),
        neo4j_pass     = config.get("NEO4J_PASSWORD")
    )

    try:
        pregunta = "¿Quién creó las herramientas en las que se inspiró FastAPI y qué dependencias o estándares utiliza?"
        print(f"=== CONSULTA VECTOR-FIRTS ===")
        print(f"Pregunta: '{pregunta}'\n")
        respuesta = pipeline.responder_pregunta(consulta_usuario=pregunta, top_k=3)

        print("\n=== PASO 3: RESPUESTA FINAL DEL LLM ===")
        print(respuesta)
    finally:
        pipeline.close()

if __name__ == "__main__":
    main()