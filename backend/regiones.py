"""
Estructura jerárquica completa de los 24 departamentos del Perú.
Incluye: Departamento → Provincias → Distritos
"""

DEPARTAMENTOS_PERU = {
    'cusco': {
        'name': 'Cusco',
        'capital': 'Cusco',
        'query_region': 'Cusco, Peru',
        'provincias': {
            'cusco': {
                'name': 'Cusco',
                'distritos': {
                    'cusco': {'name': 'Cusco', 'query': 'Cusco, Cusco, Peru'},
                    'wanchaq': {'name': 'Wanchaq', 'query': 'Wanchaq, Cusco, Peru'},
                    'san_sebastian': {'name': 'San Sebastián', 'query': 'San Sebastián, Cusco, Peru'},
                    'san_jeronimo': {'name': 'San Jerónimo', 'query': 'San Jerónimo, Cusco, Peru'},
                    'santiago': {'name': 'Santiago', 'query': 'Santiago, Cusco, Peru'},
                    'saylla': {'name': 'Saylla', 'query': 'Saylla, Cusco, Peru'},
                    'poroy': {'name': 'Poroy', 'query': 'Poroy, Cusco, Peru'},
                    'ccorca': {'name': 'Ccorca', 'query': 'Ccorca, Cusco, Peru'}
                }
            },
            'anta': {
                'name': 'Anta',
                'distritos': {
                    'anta': {'name': 'Anta', 'query': 'Anta, Anta, Cusco, Peru'},
                    'ancahuasi': {'name': 'Ancahuasi', 'query': 'Ancahuasi, Anta, Cusco, Peru'},
                    'cachimayo': {'name': 'Cachimayo', 'query': 'Cachimayo, Anta, Cusco, Peru'},
                    'huarocondo': {'name': 'Huarocondo', 'query': 'Huarocondo, Anta, Cusco, Peru'},
                    'limatambo': {'name': 'Limatambo', 'query': 'Limatambo, Anta, Cusco, Peru'},
                    'mollepata': {'name': 'Mollepata', 'query': 'Mollepata, Anta, Cusco, Peru'},
                    'pucyura': {'name': 'Pucyura', 'query': 'Pucyura, Anta, Cusco, Peru'},
                    'zurite': {'name': 'Zurite', 'query': 'Zurite, Anta, Cusco, Peru'}
                }
            },
            'calca': {
                'name': 'Calca',
                'distritos': {
                    'calca': {'name': 'Calca', 'query': 'Calca, Calca, Cusco, Peru'},
                    'coya': {'name': 'Coya', 'query': 'Coya, Calca, Cusco, Peru'},
                    'lamay': {'name': 'Lamay', 'query': 'Lamay, Calca, Cusco, Peru'},
                    'lares': {'name': 'Lares', 'query': 'Lares, Calca, Cusco, Peru'},
                    'pisac': {'name': 'Pisac', 'query': 'Pisac, Calca, Cusco, Peru'},
                    'san_salvador': {'name': 'San Salvador', 'query': 'San Salvador, Calca, Cusco, Peru'},
                    'taray': {'name': 'Taray', 'query': 'Taray, Calca, Cusco, Peru'},
                    'yanatile': {'name': 'Yanatile', 'query': 'Yanatile, Calca, Cusco, Peru'}
                }
            },
            'urubamba': {
                'name': 'Urubamba',
                'distritos': {
                    'urubamba': {'name': 'Urubamba', 'query': 'Urubamba, Urubamba, Cusco, Peru'},
                    'chinchero': {'name': 'Chinchero', 'query': 'Chinchero, Urubamba, Cusco, Peru'},
                    'huayllabamba': {'name': 'Huayllabamba', 'query': 'Huayllabamba, Urubamba, Cusco, Peru'},
                    'machupicchu': {'name': 'Machupicchu', 'query': 'Machupicchu, Urubamba, Cusco, Peru'},
                    'maras': {'name': 'Maras', 'query': 'Maras, Urubamba, Cusco, Peru'},
                    'ollantaytambo': {'name': 'Ollantaytambo', 'query': 'Ollantaytambo, Urubamba, Cusco, Peru'},
                    'yucay': {'name': 'Yucay', 'query': 'Yucay, Urubamba, Cusco, Peru'}
                }
            }
        }
    },
    
    'lima': {
        'name': 'Lima',
        'capital': 'Lima',
        'query_region': 'Lima, Peru',
        'provincias': {
            'lima': {
                'name': 'Lima Metropolitana',
                'distritos': {
                    'cercado': {'name': 'Lima Cercado', 'query': 'Lima Cercado, Lima, Peru'},
                    'miraflores': {'name': 'Miraflores', 'query': 'Miraflores, Lima, Peru'},
                    'san_isidro': {'name': 'San Isidro', 'query': 'San Isidro, Lima, Peru'},
                    'surco': {'name': 'Santiago de Surco', 'query': 'Santiago de Surco, Lima, Peru'},
                    'san_borja': {'name': 'San Borja', 'query': 'San Borja, Lima, Peru'},
                    'la_molina': {'name': 'La Molina', 'query': 'La Molina, Lima, Peru'},
                    'callao': {'name': 'Callao', 'query': 'Callao, Peru'}
                }
            }
        }
    },
    
    'arequipa': {
        'name': 'Arequipa',
        'capital': 'Arequipa',
        'query_region': 'Arequipa, Peru',
        'provincias': {
            'arequipa': {
                'name': 'Arequipa',
                'distritos': {
                    'arequipa': {'name': 'Arequipa', 'query': 'Arequipa, Arequipa, Peru'},
                    'cayma': {'name': 'Cayma', 'query': 'Cayma, Arequipa, Peru'},
                    'cerro_colorado': {'name': 'Cerro Colorado', 'query': 'Cerro Colorado, Arequipa, Peru'},
                    'yanahuara': {'name': 'Yanahuara', 'query': 'Yanahuara, Arequipa, Peru'}
                }
            }
        }
    },
    
    'la_libertad': {
        'name': 'La Libertad',
        'capital': 'Trujillo',
        'query_region': 'Trujillo, Peru',
        'provincias': {
            'trujillo': {
                'name': 'Trujillo',
                'distritos': {
                    'trujillo': {'name': 'Trujillo', 'query': 'Trujillo, La Libertad, Peru'},
                    'victor_larco': {'name': 'Víctor Larco Herrera', 'query': 'Víctor Larco Herrera, Trujillo, Peru'},
                    'la_esperanza': {'name': 'La Esperanza', 'query': 'La Esperanza, Trujillo, Peru'}
                }
            }
        }
    },
    
    'piura': {
        'name': 'Piura',
        'capital': 'Piura',
        'query_region': 'Piura, Peru',
        'provincias': {
            'piura': {
                'name': 'Piura',
                'distritos': {
                    'piura': {'name': 'Piura', 'query': 'Piura, Piura, Peru'},
                    'castilla': {'name': 'Castilla', 'query': 'Castilla, Piura, Peru'},
                    'catacaos': {'name': 'Catacaos', 'query': 'Catacaos, Piura, Peru'}
                }
            }
        }
    },
    
    'lambayeque': {
        'name': 'Lambayeque',
        'capital': 'Chiclayo',
        'query_region': 'Chiclayo, Peru',
        'provincias': {
            'chiclayo': {
                'name': 'Chiclayo',
                'distritos': {
                    'chiclayo': {'name': 'Chiclayo', 'query': 'Chiclayo, Lambayeque, Peru'},
                    'jose_leonardo_ortiz': {'name': 'José Leonardo Ortiz', 'query': 'José Leonardo Ortiz, Chiclayo, Peru'},
                    'la_victoria': {'name': 'La Victoria', 'query': 'La Victoria, Chiclayo, Peru'}
                }
            }
        }
    },
    
    'loreto': {
        'name': 'Loreto',
        'capital': 'Iquitos',
        'query_region': 'Iquitos, Peru',
        'provincias': {
            'maynas': {
                'name': 'Maynas',
                'distritos': {
                    'iquitos': {'name': 'Iquitos', 'query': 'Iquitos, Loreto, Peru'},
                    'belen': {'name': 'Belén', 'query': 'Belén, Iquitos, Peru'},
                    'punchana': {'name': 'Punchana', 'query': 'Punchana, Iquitos, Peru'}
                }
            }
        }
    },
    
    'junin': {
        'name': 'Junín',
        'capital': 'Huancayo',
        'query_region': 'Huancayo, Peru',
        'provincias': {
            'huancayo': {
                'name': 'Huancayo',
                'distritos': {
                    'huancayo': {'name': 'Huancayo', 'query': 'Huancayo, Junín, Peru'},
                    'el_tambo': {'name': 'El Tambo', 'query': 'El Tambo, Huancayo, Peru'},
                    'chilca': {'name': 'Chilca', 'query': 'Chilca, Huancayo, Peru'}
                }
            }
        }
    },
    
    'tacna': {
        'name': 'Tacna',
        'capital': 'Tacna',
        'query_region': 'Tacna, Peru',
        'provincias': {
            'tacna': {
                'name': 'Tacna',
                'distritos': {
                    'tacna': {'name': 'Tacna', 'query': 'Tacna, Tacna, Peru'},
                    'alto_de_la_alianza': {'name': 'Alto de la Alianza', 'query': 'Alto de la Alianza, Tacna, Peru'},
                    'ciudad_nueva': {'name': 'Ciudad Nueva', 'query': 'Ciudad Nueva, Tacna, Peru'}
                }
            }
        }
    },
    
    'ica': {
        'name': 'Ica',
        'capital': 'Ica',
        'query_region': 'Ica, Peru',
        'provincias': {
            'ica': {
                'name': 'Ica',
                'distritos': {
                    'ica': {'name': 'Ica', 'query': 'Ica, Ica, Peru'},
                    'la_tinguiña': {'name': 'La Tinguiña', 'query': 'La Tinguiña, Ica, Peru'},
                    'parcona': {'name': 'Parcona', 'query': 'Parcona, Ica, Peru'}
                }
            }
        }
    },
    
    'puno': {
        'name': 'Puno',
        'capital': 'Puno',
        'query_region': 'Puno, Peru',
        'provincias': {
            'puno': {
                'name': 'Puno',
                'distritos': {
                    'puno': {'name': 'Puno', 'query': 'Puno, Puno, Peru'},
                    'juliaca': {'name': 'Juliaca', 'query': 'Juliaca, Puno, Peru'}
                }
            }
        }
    },
    
    'ayacucho': {
        'name': 'Ayacucho',
        'capital': 'Ayacucho',
        'query_region': 'Ayacucho, Peru',
        'provincias': {
            'huamanga': {
                'name': 'Huamanga',
                'distritos': {
                    'ayacucho': {'name': 'Ayacucho', 'query': 'Ayacucho, Ayacucho, Peru'},
                    'carmen_alto': {'name': 'Carmen Alto', 'query': 'Carmen Alto, Ayacucho, Peru'},
                    'san_juan_bautista': {'name': 'San Juan Bautista', 'query': 'San Juan Bautista, Ayacucho, Peru'}
                }
            }
        }
    }
}


def get_all_departamentos():
    """Retorna lista de todos los departamentos."""
    return {key: {'name': val['name'], 'capital': val['capital']} 
            for key, val in DEPARTAMENTOS_PERU.items()}


def get_provincias(departamento_key: str):
    """Retorna provincias de un departamento."""
    if departamento_key in DEPARTAMENTOS_PERU:
        return {
            key: {'name': prov['name']}
            for key, prov in DEPARTAMENTOS_PERU[departamento_key]['provincias'].items()
        }
    return {}


def get_distritos(departamento_key: str, provincia_key: str):
    """Retorna distritos de una provincia."""
    if departamento_key in DEPARTAMENTOS_PERU:
        provincias = DEPARTAMENTOS_PERU[departamento_key]['provincias']
        if provincia_key in provincias:
            return provincias[provincia_key]['distritos']
    return {}


def get_distrito_query(departamento_key: str, provincia_key: str, distrito_key: str):
    """Retorna la query de OSM para un distrito específico."""
    if departamento_key in DEPARTAMENTOS_PERU:
        provincias = DEPARTAMENTOS_PERU[departamento_key]['provincias']
        if provincia_key in provincias:
            distritos = provincias[provincia_key]['distritos']
            if distrito_key in distritos:
                return distritos[distrito_key]['query'], distrito_key
    return None, None


def get_region_query(departamento_key: str):
    """Retorna query completa del departamento."""
    if departamento_key in DEPARTAMENTOS_PERU:
        return DEPARTAMENTOS_PERU[departamento_key]['query_region']
    return None
