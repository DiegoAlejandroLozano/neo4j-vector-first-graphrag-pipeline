from typing                 import Dict, Any
from langchain_openai       import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from neo4j                  import GraphDatabase

class PipelineVectorFirst:
    """
    Implementa el Pipeline Vector-First:
    1. Búsqueda semántica de entidades iniciales en Pinecone.
    2. Expansión de subgrafo contextual en Neo4j (1 o 2 Hops).
    3. Generación de respuesta enriquecida mediante LLM.
    """
    def __init__(
        self,
        openai_api_key   : str,
        neo4j_uri        : str,
        neo4j_user       : str,
        neo4j_pass       : str,
    ):
        # Modelo para vectorizar la consulta del usuario
        self.embeddings = OpenAIEmbeddings(
            api_key = openai_api_key,
            model   = "text-embedding-3-small"
        )

        # LLM para sintetizar la respuesta final
        self.llm = ChatOpenAI(
            model       = "gpt-4o-mini",
            temperature = 0,
            api_key     = openai_api_key
        )

        # Conexiones a las bases de datos
        self.neo4j_driver   = GraphDatabase.driver(
            uri  = neo4j_uri,
            auth = (neo4j_user, neo4j_pass)
        )

    def close(self):
        self.neo4j_driver.close()

    def _obtener_contexto_vector_y_grafo(
        self,
        consulta_usuario : str,
        top_k            : int = 3,
        max_hops         : int = 1
    ) -> Dict[str, Any]:
        """
        Ejecuta en una SOLA consulta Cypher:
        1. Búsqueda vectorial inicial mediante `SEARCH n in (...)`.
        2. Expansión a N saltos (`max_hops`) usando `OPTIONAL MATCH`.
        """
        vector_query = self.embeddings.embed_query(text=consulta_usuario)
        query        = f"""
        MATCH (s:Entidad)
        SEARCH s IN (
            VECTOR INDEX entity_embeddings FOR $query_vector
            LIMIT $top_k
        ) SCORE AS score

        WITH collect({{
            id    : s.id,
            nombre: s.nombre,
            text  : s.text,
            score : score
        }}) AS semillas, collect(s) AS nodos_semilla

        UNWIND nodos_semilla AS s
        OPTIONAL MATCH path = (s)-[*1..{max_hops}]->(o:Entidad)
        WITH semillas, s, path
        UNWIND (CASE WHEN path IS NULL THEN [null] ELSE relationships(path) END) AS r

        WITH semillas, s, r, endNode(r) AS obj

        RETURN 
            semillas,
            collect(DISTINCT CASE WHEN r IS NOT NULL THEN {{
                sujeto           : s.nombre,
                props_sujeto     : properties(s),
                relacion         : type(r),
                props_relacion   : properties(r),
                objeto           : obj.nombre,
                props_objeto     : properties(obj),
                etiquetas_sujeto : labels(s),
                etiquetas_objeto : labels(obj)
            }} ELSE null END) AS subgrafo_raw
        """

        with self.neo4j_driver.session() as session:
            result = session.run(query=query, query_vector=vector_query, top_k=top_k).single()

            if not result:
                return {"entidades":[], "subgrafo":[]}

            entidades = result["semillas"]
            # Filtrar valores nulos si alguna entidad semilla no tiene relaciones
            subgrafo_raw = [r for r in result["subgrafo_raw"] if r is not None]

            # Filtrar metadatos internos de Neo4j (id, nombre, vector, text)
            tripletas_contexto = []
            for r in subgrafo_raw:
                props_sub = {k: v for k, v in r["props_sujeto"].items() if k not in ["id", "nombre", "vector", "text"]}
                props_obj = {k: v for k, v in r["props_objeto"].items() if k not in ["id", "nombre", "vector", "text"]}

                tripletas_contexto.append({
                    "sujeto"           : r["sujeto"],
                    "props_sujeto"     : props_sub,
                    "relacion"         : r["relacion"],
                    "props_relacion"   : r["props_relacion"],
                    "objeto"           : r["objeto"],
                    "props_objeto"     : props_obj,
                    "etiquetas_sujeto" : r["etiquetas_sujeto"],
                    "etiquetas_objeto" : r["etiquetas_objeto"]
                })

            return {
                "entidades" : entidades,
                "subgrafo"  : tripletas_contexto
            }  
    
    def responder_pregunta(self, consulta_usuario:str, top_k: int = 3, max_hops: int = 1) -> str:
        # Obtener tanto entidades iniciales como subgrafo en un solo round-trip
        contexto = self._obtener_contexto_vector_y_grafo(
            consulta_usuario = consulta_usuario,
            top_k            = top_k,
            max_hops         = max_hops
        )

        entidades_iniciales = contexto["entidades"]
        subgrafo            = contexto["subgrafo"]

        print(f"\n[PASO 1] Entidades semánticas encontradas en Neo4j (Vector Search):")
        texto_contexto_documentos = ""
        for ent in entidades_iniciales:
            print(f" - ID: {ent['id']} | Nombre: '{ent['nombre']}' (Score: {ent['score']:.4f})")

            # Acumular el texto recuperado de las entidades
            if ent.get("text") and ent.get("text") not in texto_contexto_documentos:
                texto_contexto_documentos += f"### Entidad: {ent['nombre']}\n{ent['text']}\n\n"

        if not texto_contexto_documentos:
            texto_contexto_documentos = "No hay fragmentos de texto detallados disponibles."

        print(f"\n[PASO 2] Subgrafo contextual extraído de Neo4j ({len(subgrafo)} relaciones)")
        texto_contexto_relaciones = ""
        for i, t in enumerate(subgrafo, 1):
            lbl_sujeto = ":".join(t.get("etiquetas_sujeto", []))
            lbl_objeto = ":".join(t.get("etiquetas_objeto", []))

            str_sujeto = f"{t['sujeto']}:{lbl_sujeto}" if lbl_sujeto else t['sujeto']
            str_objeto = f"{t['objeto']}:{lbl_objeto}" if lbl_objeto else t['objeto']

            p_sub = f" {t['props_sujeto']}" if t['props_sujeto'] else ""
            p_rel = f" {t['props_relacion']}" if t['props_relacion'] else ""
            p_obj = f" {t['props_objeto']}" if t['props_objeto'] else ""

            linea = f"({str_sujeto}{p_sub}) --[{t['relacion']}{p_rel}]--> ({str_objeto}{p_obj})"
            print(f" {i:02d}. {linea}")
            texto_contexto_relaciones += f"- {linea}\n"

        if not texto_contexto_relaciones:
            texto_contexto_relaciones = "No se encontraron relaciones directas en el grafo."

        # Generar respuesta con RAG
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "Eres un asistente virtual experto respaldado por un Grafo de Conocimiento y "
                "fragmentos de documentos relevantes.\n"
                "Responde a la pregunta del usuario utilizando ÚNICAMENTE la información proporcionada "
                "en los fragmentos de texto y las relaciones estructurales del grafo. Si la información "
                "no está presente en el contexto, indícalo de manera clara."
            )),
            ("human", (
                "Pregunta del usuario: {pregunta}\n\n"
                "--- FRAGMENTOS DE TEXTO RECUPERADOS ---\n"
                "{contexto_textos}\n\n"
                "--- SUBGRAFO DE RELACIONES (Neo4j) ---\n"
                "{contexto_relaciones}\n\n"
                "Respuesta estructurada y clara:"
            ))
        ])

        chain     = prompt_template | self.llm
        respuesta = chain.invoke({
            "pregunta"            : consulta_usuario,
            "contexto_textos"     : texto_contexto_documentos,
            "contexto_relaciones" : texto_contexto_relaciones
        })

        return respuesta.content