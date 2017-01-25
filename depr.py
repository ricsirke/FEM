"""
fem(_neu) error calc sum, one point

    xpont = np.zeros(N)
    error = np.zeros(N)
    
    smallest_side = m.get_tria_smallest_side()
    print "smallest side ", smallest_side
    
    
        errors = []
        pont = [0.125, 0.5]
        for i in range(N):
            node_coords = m.nodes[i].pos.coords
            xpont[i] = pont_sol(*node_coords)
            error[i] = np.fabs(xpont[i] - x[i])
            if node_coords == pont:
                print "pont, koz, hiba: ", xpont[i], x[i], error[i]
            errors.append({"error": error[i], "pos": m.nodes[i].pos, "i":i})
            
        error_pont = [err["error"] for err in errors if err["pos"].coords == pont]
        errors_num = [err["error"] for err in errors]
        print "error in ", pont, ": ", error_pont
        print "sum of errors: ", np.sum(errors_num)
"""