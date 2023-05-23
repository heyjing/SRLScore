"""
Script exemplifying the usage of the SRLScore metric for evaluation.
"""

from SRLScore import SRLScore


if __name__ == '__main__':
    scorer = SRLScore("rouge", False)

    # This is a partial example from the QAGS-XSUM dataset.
    input_text = "London's first history day will be held on the anniversary of big ben's first day in operation. It will be first celebrated on 31 may in 2017 with celebrations and events run by historic england. The date was decided upon after a poll involving 1,000 londoners. It was closely followed by 5 september - the date of the great fire of london. The yougov questionnaire also declared the houses of parliament as the building that best sums up london. People voted for the queen as their favourite historic london hero for the moment she secretly joined the crowds to celebrate victory in europe day. The results of the poll were released to mark the launch of historic england's \"keep it london\" campaign. People were asked to select a date to celebrate the capital's history, their historic hero and the building that sums up london. Big ben's first day in operation was 31 may 1859. The campaign is intended to encourage londoners to notice, celebrate and speak up for the heritage of their city, historic england said. The public body has also launched a film entitled i am london, which celebrates the historic buildings and places that have borne witness to the capital's history. Duncan wilson, chief executive of historic england, said"
    summary_text = "Big ben's 150th anniversary has been chosen as the date to celebrate london's history."

    print(scorer.score(input_text, summary_text))