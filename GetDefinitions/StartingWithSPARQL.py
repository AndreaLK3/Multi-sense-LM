import SPARQLWrapper as SW




def get_dbpedia_def_of_word(target_word):

    sparql = SW.SPARQLWrapper("http://dbpedia.org/sparql")
    query_string = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbres: <http://dbpedia.org/resource/>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        
        SELECT DISTINCT ?encyclopedia_def
        WHERE { dbres:""" + target_word + \
        """ rdfs:comment ?encyclopedia_def 
                FILTER (LANG(?encyclopedia_def)='en')}
        """

    sparql.setQuery(query_string)
    sparql.setReturnFormat(SW.JSON)
    answer = sparql.query().convert()

    result = answer['results']['bindings'][0]['encyclopedia_def']['value']

    print(result)


get_dbpedia_def_of_word('Plant')