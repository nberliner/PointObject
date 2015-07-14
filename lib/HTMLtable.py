# -*- coding: utf-8 -*-
"""
Part of PointObject (https://github.com/nberliner/PointObject)

An IPython Notebook based anlysis tool for point-localisation
super-resolution data. 


Author(s): Niklas Berliner (niklas.berliner@gmail.com)

Copyright (C) 2015 Niklas Berliner
"""


class HTMLtable(object):
    """
    Generates a HTML table for the use in iPython Notebooks.
    Input are two lists with the row and column labels respectively.
    
    @author: Niklas Berliner
    @email:  niklas.berliner@gmail.com
    """
    
    def __init__(self, precision=2):
        """ precision: number of decimal positions to be shown """
        self.precision = precision
    
    def __call__(self, rows, columns, header=None):
        """
        Input:
          rows:     A list containing the row labels
          columns:  A list containing the column labels
          header:   A header for the table (optional)
        
        Output
          table:    The string of the HTML table of the form
                  
                  | columTitle1 | columTitle2 | columTitle2 | etc. |
                  --------------------------------------------------
                  | rowTitle1   |     %.2f    |    %.2f     | etc. |
                  | rowTitle2   |     %.2f    |    %.2f     | etc. |
                  | rowTitle3   |     %.2f    |    %.2f     | etc. |
                  etc.
                  
                  where precision was set to 2
        
        Important
          You can display the HTML table in iPyhton notebook by
          
            from IPython.display import HTML
            htmlTable = HTMLtable()
            HTML( htmlTable( ['row1','row2','row3',], ['col1','col2','col3',] ) % results )
        
          where `results` must be a tuple of floats of length (len(rows)-1)*(len(columns)-1)
          to match the number of ``%.2f`s in the table.
        """
        
        assert( isinstance(rows,    list) )
        assert( isinstance(columns, list) )
        
        table = self._makeTable(rows, columns)
        
        if header == None:
            return table
        else:
            return "<h1>" + str(header) + "</h1>\n" + table
    
    def _makeTable(self, rows, columns):
        # Start the table
        table = "<table>\n"
        
        # Generate the first row as header
        table += "<tr>\n"
        for c in range(len(columns)):
            table += "  <th><div align = \"center\">" + str(columns[c]) + "</th>\n"
        table += "</tr>\n"
        
        # Generate the subsequent ows for the data values
        for r in range(len(rows)):
            table += "<tr>\n"
            for c in range(len(columns)):
                if c == 0: # add the row label
                    table += "  <td><div align = \"right\"><b>" + str(rows[r]) + "</b></td>\n"
                else:
                    table += "  <td><div align = \"right\">%0." + str(self.precision) + "f</div></td>\n"
            table += "</tr>\n"
        
        table += "</table>"

        return table


if __name__ == '__main__':
    
    # Quick example
    # Requires python3
    
    rows = [1,2,3,4,5]
    columns = ['frame', 'curvature', 'something else']
    
    html = HTMLtable()
    
    print(html(rows, columns, 'This is the header'))