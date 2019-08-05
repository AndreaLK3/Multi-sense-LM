import SPARQLWrapper as SW
import logging
import Utils



# Note: all the resources at dbres start with upper case. e.g. Sea, Plant. We modify the target word accordingly
# Space == underscore, e.g. New_York
def retrieve_dbpedia_def(target_word):

    upcaseStart_target_word = target_word[0].upper() + target_word[1:]
    logging.info("*** DBpedia : " + target_word + ' ...')

    sparql = SW.SPARQLWrapper("http://dbpedia.org/sparql")
    query_string = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbres: <http://dbpedia.org/resource/>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        
        SELECT DISTINCT ?encyclopedia_def
        WHERE { dbres:""" + upcaseStart_target_word + \
        """ rdfs:comment ?encyclopedia_def 
                FILTER (LANG(?encyclopedia_def)='en')}
        """

    sparql.setQuery(query_string)
    sparql.setReturnFormat(SW.JSON)
    answer = sparql.query().convert()
    logging.info(answer)
    result = answer['results']['bindings'][0]['encyclopedia_def']['value']
    return result


def main():
    Utils.init_logging("DBpedia.log", logging.INFO)

    ency_def = retrieve_dbpedia_def('Plant')
    logging.info(ency_def)

#main()