import os
import glob
import shutil
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
import os
import datetime
import array
def matmodel2specfem(vp,vs,rho,dx,dz):

    # determine grid spacing
    nx, nz = vp.shape

    # original model range
    x_range = [0.0, nx*dx-1]   # x in m
    z_range = [0.0, nz*dz-1]   # depth in m


    x = np.linspace(x_range[0], x_range[1], nx)
    z = np.linspace(z_range[0], z_range[1], nz)



    model = {'vp': vp, 'vs': vs, 'rho': rho,
              'x': x, 'z': z, 'dx': dx, 'dz': dz}


    # gets minimum velocity for elastic domain (vs != 0)
    v_min = vs[np.where(vs > 0.0)].min()   # Vs
    if v_min == 0.0: v_min = vp.min()
    # maximum grid spacing
    d_max = max(dx,dz)
    MINIMUM_PTS_PER_WAVELENGTH = 5 
    # maximum frequency resolved (using a minimum number of points per wavelength)
    f_max = v_min / (MINIMUM_PTS_PER_WAVELENGTH * d_max)

    # stats
    print("    vp  min/max = {:8.2f} / {:8.2f} (m/s)".format(vp.min(),vp.max()))
    print("    vs  min/max = {:8.2f} / {:8.2f} (m/s)".format(vs.min(),vs.max()))
    print("    rho min/max = {:8.2f} / {:8.2f} (kg/m^3)".format(rho.min(),rho.max()))
    print("")
    print("  grid: x     range min/max = {:.2f} / {:.2f} (m)".format(x.min(),x.max()))
    print("        depth range min/max = {:.2f} / {:.2f} (m)".format(z.min(),z.max()))
    print("        nx / nz             = {} / {}".format(nx,nz))
    print("        dx / dz             = {:.2f} / {:.2f} (m)".format(dx,dz))
    print("")
    print("        resolved maximum frequency = ",f_max)
    print("")

    # plot model
    plt.figure(figsize=(10,6),dpi=150)
    plt.title("Marmousi2 - Vs original")
    plt.pcolormesh(model['x'][::1], model['z'][::1], model['vs'][::1,::1].T, shading='auto', cmap="RdYlBu_r")
    #plt.axis("scaled")
    plt.gca().invert_yaxis()
    plt.colorbar(shrink=0.3)

    # saves as JPEG file
    filename = "vp" + ".jpg"
    plt.savefig(filename)
    print("  plotted as ",filename)
    print("")
    
    return model
    # if show_plot:
    #     plt.show()
def create_tomography_file(model,filename,use_tomography_file_ascii_format):
    """
    creates an ascii tomography file in SPECFEM2D format
    """
    print("creating tomography file:")

    # initializes header variables
    header_origin_x = sys.float_info.max
    header_origin_z = sys.float_info.max

    header_end_x = -sys.float_info.max
    header_end_z = -sys.float_info.max

    header_vp_min = sys.float_info.max
    header_vp_max = -sys.float_info.max

    header_vs_min = sys.float_info.max
    header_vs_max = -sys.float_info.max

    header_rho_min = sys.float_info.max
    header_rho_max = -sys.float_info.max

    # model dimensions
    nx, nz = model['vp'].shape

    # header infos
    header_dx = model['dx']        # increment in x-direction
    header_dz = model['dz']        # increment in x-direction

    header_nx = nx   # x-direction
    header_nz = nz   # z-direction

    print("  dimension:")
    print("    nx / nz = {} / {}".format(header_nx,header_nz))
    print("    dx / dz = {} / {} (m)".format(header_dx,header_dz))

    # note: the current model arrays have x/z coordinates with z being depth.
    #       for SPECFEM2D, we need instead Z-coordinates where z-coordinates would point positive upwards.
    #       thus, we will reverse the looping direction and loop to go bottom-up, instead of top-down.
    ztop_model = model['z'].max()

    # collect model data
    if use_tomography_file_ascii_format:
        output_data = list()
    else:
        output_data_array = np.empty([5,nx*nz],dtype='f')
        iline = 0

    # loops over z
    for j in range(nz):
        # loops over x (must have inner loop over x)
        for i in range(nx):
            # reverse indexing direction to bottom-up
            k = nz - 1 - j

            # position
            x_val = model['x'][i]
            depth_val = model['z'][k]

            # will need to flip z-values to have z-direction positive up
            # from:     0.0 == top-water   , 3500.0 == bottom-layer of model
            # to  :  3500.0 == top-layer   ,    0.0 == bottom-layer of model
            z_val = ztop_model - depth_val

            # model parameters
            vp_val = model['vp'][i,k]
            vs_val = model['vs'][i,k]
            rho_val = model['rho'][i,k]

            # data line format:
            #x #z #vp #vs #density (see in src/specfem2d/define_external_model_from_tomo_file.f90)
            if use_tomography_file_ascii_format:
                # ascii format output
                output_data.append("{} {} {} {} {}\n".format(x_val,z_val,vp_val,vs_val,rho_val))
            else:
                # binary format
                output_data_array[:,iline] = [x_val,z_val,vp_val,vs_val,rho_val]
                iline += 1

            # header stats
            header_origin_x = min(header_origin_x,x_val)
            header_origin_z = min(header_origin_z,z_val)

            header_end_x = max(header_end_x,x_val)
            header_end_z = max(header_end_z,z_val)

            header_vp_min = min(header_vp_min,vp_val)
            header_vp_max = max(header_vp_max,vp_val)

            header_vs_min = min(header_vs_min,vs_val)
            header_vs_max = max(header_vs_max,vs_val)

            header_rho_min = min(header_rho_min,rho_val)
            header_rho_max = max(header_rho_max,rho_val)

    print("    x origin / end = {} / {} (m)".format(header_origin_x,header_end_x))
    print("    z origin / end = {} / {} (m)".format(header_origin_z,header_end_z))
    print("")

    print("  tomographic model statistics:")
    print("    vp  min/max : {:8.2f} / {:8.2f} (m/s)".format(header_vp_min,header_vp_max))
    print("    vs  min/max : {:8.2f} / {:8.2f} (m/s)".format(header_vs_min,header_vs_max))
    print("    rho min/max : {:8.2f} / {:8.2f} (kg/m3)".format(header_rho_min,header_rho_max))
    print("")

    if header_vp_min <= 0.0:
        print("WARNING: Vp has invalid entries with a minimum of zero!")
        print("         The provided output model is likely invalid.")
        print("         Please check with your inputs...")
        print("")

    if header_vs_min <= 0.0:
        print("WARNING: Vs has entries with a minimum of zero!")
        print("         The provided output model is likely invalid.")
        print("         Please check with your inputs...")
        print("")

    if header_rho_min <= 0.0:
        print("WARNING: Density has entries with a minimum of zero!")
        print("         The provided output model is likely invalid.")
        print("         Please check with your inputs...")
        print("")

    # data header
    data_header = list()
    data_header.append("# tomography model - zhangchang modified zhangchang23@malis.jlu.edu.cn\n")
    data_header.append("#\n")

#     # providence
#     data_header.append("# providence\n")
#     data_header.append("# created             : {}\n".format(str(datetime.datetime.now())))
#     data_header.append("# command             : {}\n".format(" ".join(sys.argv)))
#     data_header.append("#\n")

    # tomographic model format
    data_header.append("# model format\n")
    data_header.append("# model type          : Marmousi2\n")

    data_header.append("# coordinate format   : x / z  # z-direction (positive up)\n")
    data_header.append("#\n")

    # tomography model header infos
    #origin_x #origin_y #origin_z #end_x #end_y #end_z          - start-end dimensions
    data_header.append("#origin_x #origin_z #end_x #end_z\n")
    data_header.append("{} {} {} {}\n".format(header_origin_x,header_origin_z,header_end_x,header_end_z))
    #dx #dy #dz                                                 - increments
    data_header.append("#dx #dz\n")
    data_header.append("{} {}\n".format(header_dx,header_dz))
    #nx #ny #nz                                                 - number of models entries
    data_header.append("#nx #nz\n")
    data_header.append("{} {}\n".format(header_nx,header_nz))
    #vp_min #vp_max #vs_min #vs_max #density_min #density_max   - min/max stats
    data_header.append("#vp_min #vp_max #vs_min #vs_max #density_min #density_max\n")
    data_header.append("{} {} {} {} {} {}\n".format(header_vp_min,header_vp_max,header_vs_min,header_vs_max,header_rho_min,header_rho_max))
    data_header.append("#\n")

    # data record info
    data_header.append("# data records - format:\n")
    # format: x #z #vp #vs #density
    data_header.append("#x #z #vp #vs #density\n")

    ## SPECFEM2D tomographic model format
    if use_tomography_file_ascii_format:
        ## file header
        print("  using ascii file format")
        print("")

        ## writes output file
        os.system("mkdir -p DATA")

        with open(filename, "w") as fp:
            fp.write(''.join(data_header))
            fp.write(''.join(output_data))
            fp.close()

    else:
        # binary format
        print("  using binary file format")
        print("")
        
        # collects header info
        output_header_data = np.array([ header_origin_x, header_origin_z, header_end_x, header_end_z, \
                                        header_dx, header_dz, \
                                        header_nx, header_nz, \
                                        header_vp_min, header_vp_max, header_vs_min, header_vs_max, header_rho_min, header_rho_max ])

        ## writes output file (in binary format)
        os.system("mkdir -p DATA")
        filename = filename + '.bin'
        with open(filename, "wb") as fp:
            # header
            print("    header:")
            write_binary_file_custom_real_array(fp,output_header_data)
            # model data
            print("    model data:")
            write_binary_file_custom_real_array(fp,output_data_array)
            fp.close()

        # use data header as additional metadata infos
        data_header.append("#\n")
        data_header.append("# data file: {}\n".format(filename))
        data_header.append("#\n")

        filename_info = filename + '.info'
        with open(filename_info, "w") as fp:
            fp.write(''.join(data_header))
            fp.close()

    print("tomography model written to: ",filename)
    print("")


#
#----------------------------------------------------------------------------
#


def write_binary_file_custom_real_array(file,data):
    """
    writes data array to binary file
    """
    # defines float 'f' or double precision 'd' for binary values
    custom_type = 'f'

    # gets array length in bytes
    # marker
    binlength = array.array('i')
    num_points = data.size
    if custom_type == 'f':
        # float (4 bytes) for single precision
        binlength.fromlist([num_points * 4])
    else:
        # double precision
        binlength.fromlist([num_points * 8])

    # user output
    print("    array length = ",binlength," Bytes")
    print("    number of points in array = ",num_points)
    print("    memory required: in (kB) {:.4f} / in (MB): {:.4f}".format(binlength[0] / 1024.0, binlength[0] / 1024.0 / 1024.0))
    print("")

    # writes array data
    binvalues = array.array(custom_type)

    data = np.reshape(data, (num_points), order='F') # fortran-like index ordering
    #print("debug: data ",data.tolist())

    binvalues.fromlist(data.tolist()) #.tolist())

    # fortran binary file: file starts with array-size again
    binlength.tofile(file)
    # data
    binvalues.tofile(file)
    # fortran binary file: file ends with array-size again
    binlength.tofile(file)