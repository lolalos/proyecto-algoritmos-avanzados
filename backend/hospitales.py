"""
Base de datos de hospitales importantes de Perú organizados por región.
Incluye coordenadas GPS para cálculo de rutas de emergencia.
"""

HOSPITALES_PERU = {
    'cusco': [
        {
            'name': 'Hospital Regional Cusco',
            'lat': -13.522472,
            'lon': -71.967461,
            'tipo': 'Regional',
            'nivel': 'III-1',
            'direccion': 'Av. de la Cultura S/N, Wanchaq, Cusco 08003'
        },
        {
            'name': 'Hospital Antonio Lorena',
            'lat': -13.519437,
            'lon': -71.964750,
            'tipo': 'General',
            'nivel': 'II-2',
            'direccion': 'Av. de la Cultura 720, Plaza Túpac Amaru, Wanchaq, Cusco 08003'
        },
        {
            'name': 'Hospital Adolfo Guevara Velasco (EsSalud)',
            'lat': -13.518803,
            'lon': -71.964425,
            'tipo': 'Seguridad Social',
            'nivel': 'III-1',
            'direccion': 'Av. de la Cultura 705, Wanchaq, Cusco 08003'
        },
        {
            'name': 'Clínica Pardo',
            'lat': -13.524489,
            'lon': -71.975198,
            'tipo': 'Privada',
            'nivel': 'III-1',
            'direccion': 'Av. de la Cultura 710, Cusco 08000'
        },
        {
            'name': 'Hospital de Contingencia Solidaridad',
            'lat': -13.531662,
            'lon': -71.978771,
            'tipo': 'Contingencia',
            'nivel': 'II-2',
            'direccion': 'Av. Circunvalación 1958, Santiago, Cusco 08000'
        }
    ],
    
    'lima': [
        {
            'name': 'Hospital Rebagliati (EsSalud)',
            'lat': -12.0879,
            'lon': -77.0501,
            'tipo': 'Seguridad Social',
            'nivel': 'III-2'
        },
        {
            'name': 'Hospital Almenara (EsSalud)',
            'lat': -12.0587,
            'lon': -77.0359,
            'tipo': 'Seguridad Social',
            'nivel': 'III-2'
        },
        {
            'name': 'Hospital Dos de Mayo',
            'lat': -12.0567,
            'lon': -77.0433,
            'tipo': 'General',
            'nivel': 'III-1'
        },
        {
            'name': 'Hospital Arzobispo Loayza',
            'lat': -12.0608,
            'lon': -77.0493,
            'tipo': 'General',
            'nivel': 'III-1'
        },
        {
            'name': 'Hospital Cayetano Heredia',
            'lat': -12.0022,
            'lon': -77.0509,
            'tipo': 'Especializado',
            'nivel': 'III-1'
        },
        {
            'name': 'Hospital Edgardo Rebagliati Martins',
            'lat': -12.0879,
            'lon': -77.0501,
            'tipo': 'Seguridad Social',
            'nivel': 'III-2'
        }
    ],
    
    'arequipa': [
        {
            'name': 'Hospital Regional Honorio Delgado',
            'lat': -16.3978,
            'lon': -71.5372,
            'tipo': 'Regional',
            'nivel': 'III-1'
        },
        {
            'name': 'Hospital Goyeneche',
            'lat': -16.4067,
            'lon': -71.5369,
            'tipo': 'General',
            'nivel': 'III-1'
        },
        {
            'name': 'Hospital Carlos Alberto Seguín Escobedo (EsSalud)',
            'lat': -16.4100,
            'lon': -71.5250,
            'tipo': 'Seguridad Social',
            'nivel': 'III-1'
        }
    ],
    
    'trujillo': [
        {
            'name': 'Hospital Regional Docente de Trujillo',
            'lat': -8.1166,
            'lon': -79.0289,
            'tipo': 'Regional',
            'nivel': 'III-1'
        },
        {
            'name': 'Hospital Belén de Trujillo',
            'lat': -8.1089,
            'lon': -79.0321,
            'tipo': 'General',
            'nivel': 'II-2'
        },
        {
            'name': 'Hospital Victor Lazarte Echegaray (EsSalud)',
            'lat': -8.1050,
            'lon': -79.0380,
            'tipo': 'Seguridad Social',
            'nivel': 'III-1'
        }
    ],
    
    'piura': [
        {
            'name': 'Hospital Cayetano Heredia',
            'lat': -5.1945,
            'lon': -80.6328,
            'tipo': 'Regional',
            'nivel': 'III-1'
        },
        {
            'name': 'Hospital Santa Rosa',
            'lat': -5.1968,
            'lon': -80.6364,
            'tipo': 'General',
            'nivel': 'II-2'
        },
        {
            'name': 'Hospital Jorge Reategui Delgado (EsSalud)',
            'lat': -5.1920,
            'lon': -80.6290,
            'tipo': 'Seguridad Social',
            'nivel': 'III-1'
        }
    ],
    
    'chiclayo': [
        {
            'name': 'Hospital Regional Lambayeque',
            'lat': -6.7714,
            'lon': -79.8391,
            'tipo': 'Regional',
            'nivel': 'III-1'
        },
        {
            'name': 'Hospital Almanzor Aguinaga Asenjo (EsSalud)',
            'lat': -6.7670,
            'lon': -79.8380,
            'tipo': 'Seguridad Social',
            'nivel': 'III-1'
        }
    ],
    
    'iquitos': [
        {
            'name': 'Hospital Regional de Loreto',
            'lat': -3.7437,
            'lon': -73.2516,
            'tipo': 'Regional',
            'nivel': 'II-2'
        },
        {
            'name': 'Hospital Apoyo Iquitos',
            'lat': -3.7492,
            'lon': -73.2534,
            'tipo': 'Apoyo',
            'nivel': 'II-1'
        }
    ],
    
    'huancayo': [
        {
            'name': 'Hospital Regional Docente Clínico Quirúrgico Daniel Alcides Carrión',
            'lat': -12.0689,
            'lon': -75.2099,
            'tipo': 'Regional',
            'nivel': 'III-1'
        },
        {
            'name': 'Hospital El Carmen',
            'lat': -12.0650,
            'lon': -75.2040,
            'tipo': 'General',
            'nivel': 'II-2'
        }
    ],
    
    'tacna': [
        {
            'name': 'Hospital Hipólito Unanue',
            'lat': -18.0067,
            'lon': -70.2456,
            'tipo': 'Regional',
            'nivel': 'III-1'
        }
    ],
    
    'ica': [
        {
            'name': 'Hospital Regional de Ica',
            'lat': -14.0678,
            'lon': -75.7286,
            'tipo': 'Regional',
            'nivel': 'II-2'
        },
        {
            'name': 'Hospital Félix Torrealva Gutiérrez (EsSalud)',
            'lat': -14.0701,
            'lon': -75.7250,
            'tipo': 'Seguridad Social',
            'nivel': 'III-1'
        }
    ],
    
    'puno': [
        {
            'name': 'Hospital Regional Manuel Núñez Butrón',
            'lat': -15.8402,
            'lon': -70.0219,
            'tipo': 'Regional',
            'nivel': 'II-2'
        }
    ],
    
    'ayacucho': [
        {
            'name': 'Hospital Regional de Ayacucho',
            'lat': -13.1631,
            'lon': -74.2236,
            'tipo': 'Regional',
            'nivel': 'II-2'
        }
    ]
}


def get_hospitales_region(region_key: str):
    """Retorna lista de hospitales de una región."""
    return HOSPITALES_PERU.get(region_key, [])


def get_all_regions_with_hospitals():
    """Retorna lista de regiones que tienen hospitales definidos."""
    return list(HOSPITALES_PERU.keys())


def count_hospitales():
    """Retorna estadísticas de hospitales por región."""
    stats = {}
    for region, hospitales in HOSPITALES_PERU.items():
        stats[region] = {
            'total': len(hospitales),
            'tipos': {}
        }
        for hosp in hospitales:
            tipo = hosp.get('tipo', 'Desconocido')
            stats[region]['tipos'][tipo] = stats[region]['tipos'].get(tipo, 0) + 1
    return stats
