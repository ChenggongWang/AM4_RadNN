module radiation_driver_nn_mod

! <OVERVIEW>
!    radiation_driver_nn_mod provides a neural network version of
!    radiaitive transfer module to the radiation driver.
! </OVERVIEW>
use fms_mod,               only: FATAL, NOTE, error_mesg, stdout
use fms2_io_mod,           only: FmsNetcdfFile_t, read_data,&
                                 open_file, close_file
use diag_manager_mod,      only: register_diag_field, send_data, &
                                 diag_manager_init
use time_manager_mod,      only: time_manager_init, time_type, operator(>)
use radiation_driver_types_mod, only: radiation_control_type, &
                                      astronomy_type, &
                                      rad_output_type
use physics_radiation_exch_mod,only: clouds_from_moist_block_type
use radiative_gases_types_mod, only: radiative_gases_type
use shortwave_driver_mod, only: get_solar_constant
!--------------------------------------------------------------------

implicit none
private
! cgw: for NN_diag
integer :: idtlev, idplay, idtlay, idh2o, ido3, idzlay, idsd, &
           idsf, idsl, idsi, idcd, idcf, idcl, idci, &
           idps, idts, idzen, &
           idvdir, idvdif, ididir, ididif, idfrac, idland, ider,idsolarfact, idsolar, &
           id_nn_tdt_lw, id_nn_tdt_sw, &
           id_nn_lwdn_sfc, id_nn_lwup_sfc, id_nn_swdn_sfc, id_nn_swup_sfc, &
           id_nn_swdn_toa, id_nn_swup_toa, id_nn_olr, &
           id_nn_tdt_lw_clr,id_nn_tdt_sw_clr, &
           id_nn_lwdn_sfc_clr, id_nn_swdn_sfc_clr, id_nn_swup_sfc_clr, &
           id_nn_swup_toa_clr, id_nn_olr_clr 
character(len=32) :: mod_name="NN_diag"

public  radiation_driver_nn_init, &
        produce_rad_nn_diag,&
        test
!        nn_radiation_calc, &

!----------------------------------------------------------------------
! cgw
! define type for linear layer
! contains weight(2d) and bias(1d)
!----------------------------------------------------------------------
public NN_Linear_layer_type
type :: NN_Linear_layer_type 
    real(kind=4), dimension(:,:), pointer :: weight=>NULL()
    real(kind=4), dimension(:),   pointer :: bias=>NULL()
end type NN_Linear_layer_type
public NN_FC_type
type :: NN_FC_type
    integer :: num_hid_nodes
    integer :: num_layers
    type(NN_Linear_layer_type), dimension(:), pointer:: Layers
end type NN_FC_type

                         contains

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
!                     PUBLIC SUBROUTINES
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subroutine radiation_driver_nn_init(do_rad_nn, axes, time, rad_nn_para_nc, Rad_NN_FC)
    logical,                     intent(in)    :: do_rad_nn
    type(time_type),             intent(in)    :: time
    integer, dimension(4),       intent(in)    :: axes
    character(len=32), intent(in) :: rad_nn_para_nc
    type(NN_FC_type),  dimension(:), intent(inout) :: Rad_NN_FC
    
    ! ------ local -------
    character(len=32) :: fmt_str
    type(FmsNetcdfFile_t)       :: Rad_NN_para_fileobj        !< Fms2_io fileobj
    character(len=100), dimension(4) :: rad_nn_para_nc_4filepath
    integer :: nn_size0, nn_size1, inn, outunit
    integer :: nn_num_layers, ilayer
    integer, dimension(4) :: a
    
    outunit = stdout()
    if (do_rad_nn) then
        call error_mesg ('radiation_driver_mod',  &
                 'Initializing the rad_nn module', NOTE)
    
        call error_mesg ('radiation_driver_mod',  &
                 'Reading NetCDF file to obtain NN parameters. INPUT/'//trim(rad_nn_para_nc), NOTE)
        rad_nn_para_nc_4filepath(1) = "INPUT/"//trim(rad_nn_para_nc)//"lw_csaf_Li5Relu_EY.nc"
        rad_nn_para_nc_4filepath(2) = "INPUT/"//trim(rad_nn_para_nc)//"lw_af_Li5Relu_EY.nc"
        rad_nn_para_nc_4filepath(3) = "INPUT/"//trim(rad_nn_para_nc)//"sw_csaf_Li5Relu_EY.nc"
        rad_nn_para_nc_4filepath(4) = "INPUT/"//trim(rad_nn_para_nc)//"sw_af_Li5Relu_EY.nc"
        
        ! read para file  
        do inn = 1, 4
            if (open_file(Rad_NN_para_fileobj, rad_nn_para_nc_4filepath(inn), "read" )) then
                ! read num of NN layers
                call read_data(Rad_NN_para_fileobj, 'LN', nn_num_layers)
    
                Rad_NN_FC(inn)%num_layers = nn_num_layers
                allocate(Rad_NN_FC(inn)%Layers(Rad_NN_FC(inn)%num_layers))
                ! read weight and bias for each layer
                do ilayer = 1, nn_num_layers 
                    ! read size of each layer
                    write (fmt_str, "(A4,I1,I1)") "size", ilayer, 0
                    call read_data(Rad_NN_para_fileobj, fmt_str, nn_size0)
                    write (fmt_str, "(A4,I1,I1)") "size", ilayer, 1
                    call read_data(Rad_NN_para_fileobj, fmt_str, nn_size1)
                    allocate(Rad_NN_FC(inn)%Layers(ilayer)%weight(nn_size1,nn_size0)) !fortran reverse order
                    allocate(Rad_NN_FC(inn)%Layers(ilayer)%bias(nn_size0))
                    write(outunit,*) 'nn_size', nn_size0, nn_size1
                    write (fmt_str, "(A1,I1)") "W", ilayer
                    call read_data(Rad_NN_para_fileobj, fmt_str, Rad_NN_FC(inn)%Layers(ilayer)%weight(:,:))
                    write (fmt_str, "(A1,I1)") "B", ilayer
                    call read_data(Rad_NN_para_fileobj, fmt_str, Rad_NN_FC(inn)%Layers(ilayer)%bias(:))
                    ! for diagnose purpose
                    write(outunit,*) Rad_NN_FC(inn)%Layers(ilayer)%bias 
                end do 
                Rad_NN_FC(inn)%num_hid_nodes = nn_size1
                call close_file(Rad_NN_para_fileobj)
            else
                call error_mesg ('radiation_driver_mod',  &
                     'rad_nn_para_nc file open failed. '//trim(rad_nn_para_nc_4filepath(inn)), FATAL)
            end if
        end do
    
        !write(outunit,*) 'register output mod: ', mod_name
        a(1:2) = axes(1:2)
        ! pfull
        a(3) = axes(3)
    
        id_nn_tdt_lw = register_diag_field(mod_name, "nn_tdt_lw", a(1:3), time,  "nn_tdt_lw", "K/s")
        id_nn_tdt_sw = register_diag_field(mod_name, "nn_tdt_sw", a(1:3), time,  "nn_tdt_sw", "K/s")
        id_nn_tdt_lw_clr = register_diag_field(mod_name, "nn_tdt_lw_clr", a(1:3), time,  "nn_tdt_lw_clr", "K/s")
        id_nn_tdt_sw_clr = register_diag_field(mod_name, "nn_tdt_sw_clr", a(1:3), time,  "nn_tdt_sw_clr", "K/s")
    
    
        id_nn_lwdn_sfc = register_diag_field(mod_name, "nn_lwdn_sfc", a(1:2), time,  "nn_lwdn_sfc", "Wm-2")
        id_nn_lwup_sfc = register_diag_field(mod_name, "nn_lwup_sfc", a(1:2), time,  "nn_lwup_sfc", "Wm-2")
        id_nn_swdn_sfc = register_diag_field(mod_name, "nn_swdn_sfc", a(1:2), time,  "nn_swdn_sfc", "Wm-2")
        id_nn_swup_sfc = register_diag_field(mod_name, "nn_swup_sfc", a(1:2), time,  "nn_swup_sfc", "Wm-2")
        id_nn_swdn_toa = register_diag_field(mod_name, "nn_swdn_toa", a(1:2), time,  "nn_swdn_toa", "Wm-2")
        id_nn_swup_toa = register_diag_field(mod_name, "nn_swup_toa", a(1:2), time,  "nn_swup_toa", "Wm-2")
        id_nn_olr      = register_diag_field(mod_name, "nn_olr", a(1:2), time,  "nn_olr", "Wm-2")
        id_nn_lwdn_sfc_clr = register_diag_field(mod_name, "nn_lwdn_sfc_clr", a(1:2), time,  "nn_lwdn_sfc_clr", "Wm-2")
        id_nn_swdn_sfc_clr = register_diag_field(mod_name, "nn_swdn_sfc_clr", a(1:2), time,  "nn_swdn_sfc_clr", "Wm-2")
        id_nn_swup_sfc_clr = register_diag_field(mod_name, "nn_swup_sfc_clr", a(1:2), time,  "nn_swup_sfc_clr", "Wm-2")
        id_nn_swup_toa_clr = register_diag_field(mod_name, "nn_swup_toa_clr", a(1:2), time,  "nn_swup_toa_clr", "Wm-2")
        id_nn_olr_clr      = register_diag_field(mod_name, "nn_olr_clr", a(1:2), time,  "nn_olr_clr", "Wm-2")
    
    end if ! do_rad_nn
    ! diag
    !write(outunit,*) 'register output mod: ', mod_name
    a(1:2) = axes(1:2)
    a(3) = axes(4)
    !3D atmosphere fields.
    ! phalf
    idtlev = register_diag_field(mod_name, "tflux", a(1:3), time, &
                                 "level temperature", "K")
    
    ! pfull
    a(3) = axes(3)
    idtlay = register_diag_field(mod_name, "ta", a(1:3), time, &
                                 "layer temperature", "K")
    idzlay = register_diag_field(mod_name, "layer_thickness", a(1:3), time, &
                                 "layer_thickness", "m")
    idh2o = register_diag_field(mod_name, "water_vapor", a(1:3), time, &
                                "water_vapor", "kg water / kg dry")
    ido3 = register_diag_field(mod_name, "ozone", a(1:3), time, &
                             "ozone", "mol mol-1")
    idsd = register_diag_field(mod_name, "stratiform_droplet_number", a(1:3), time, &
                               "stratiform_droplet_number", "")
    idsf = register_diag_field(mod_name, "stratiform_cloud_fraction", a(1:3), time, &
                               "stratiform_cloud_fraction", "none")
    idsl = register_diag_field(mod_name, "stratiform_liquid_content", a(1:3), time, &
                               "stratiform_liquid_content", "")
    idsi = register_diag_field(mod_name, "stratiform_ice_content", a(1:3), time, &
                               "stratiform_ice_content", "")
    idcd = register_diag_field(mod_name, "shallow_droplet_number", a(1:3), time, &
                               "shallow_droplet_number", "")
    idcf = register_diag_field(mod_name, "shallow_cloud_fraction", a(1:3), time, &
                               "shallow_cloud_fraction", "none")
    idcl = register_diag_field(mod_name, "shallow_liquid_content", a(1:3), time, &
                               "shallow_liquid_content", "")
    idci = register_diag_field(mod_name, "shallow_ice_content", a(1:3), time, &
                               "shallow_ice_content", "")
    !2D fields (non-vertical)
    idps = register_diag_field(mod_name, "ps", a(1:2), time, &
                                 "surface pressure", "Pa")
    idts = register_diag_field(mod_name, "ts", a(1:2), time, &
                               "surface temperature", "K")
    idvdir = register_diag_field(mod_name, "visible_direct_albedo", a(1:2), time, &
                                 "visible_direct_albedo", "none")
    idvdif = register_diag_field(mod_name, "visible_diffuse_albedo", a(1:2), time, &
                                 "visible_diffuse_albedo", "none")
    ididir = register_diag_field(mod_name, "infrared_direct_albedo", a(1:2), time, &
                                 "infrared_direct_albedo", "none")
    ididif = register_diag_field(mod_name, "infrared_diffuse_albedo", a(1:2), time, &
                                 "infrared_diffuse_albedo", "none")
    idsolarfact = register_diag_field(mod_name, "solarfactor", a(1:2), time, &
                                 "solarfactor", "none")
    idzen = register_diag_field(mod_name, "cosz", a(1:2), time, &
                                "cosz", "none")
    idfrac = register_diag_field(mod_name, "dayf", a(1:2), time, &
                               "dayf", "none")
    !1D 
    idsolar = register_diag_field(mod_name, "solar_constant", time, &
                                  "solar_constant", "none")
    ! earth_sun_distance_fraction
    ider = register_diag_field(mod_name, "rrsun", time, &
                               "rrsun", "none")

end subroutine radiation_driver_nn_init

subroutine produce_rad_nn_diag(time_next, is, js, &
                               phalf, temp, tflux,  tsfc, rh2o, Rad_gases, Astro, solar_constant_used, &
                               asfc_vis_dir, asfc_vis_dif, asfc_nir_dir, asfc_nir_dif, &
                               moist_clouds_block, &
                               tdt_sw, tdt_lw, &
                               lwdn_sfc, lwup_sfc, swdn_sfc, swup_sfc,  &
                               swdn_toa, swup_toa, olr, & 
                               tdt_sw_clr, tdt_lw_clr, &
                               lwdn_sfc_clr, swdn_sfc_clr, swup_sfc_clr, &
                               swup_toa_clr, olr_clr  )
    !--------------------------------------------------------------------
    type(time_type),              intent(in)        :: time_next
    integer,                      intent(in)        :: is, js
    real, dimension(:,:,:),       intent(in)        :: phalf, temp, &
                                                       tflux, rh2o
    real, dimension(:,:),         intent(in)        :: tsfc
    type(radiative_gases_type),   intent(in)        :: Rad_gases
    type(astronomy_type),         intent(in)        :: Astro
    real,                         intent(in)        :: solar_constant_used
    real, dimension(:,:),         intent(in)        :: asfc_vis_dir, &
                                                       asfc_nir_dir, &
                                                       asfc_vis_dif, &
                                                       asfc_nir_dif
    type(clouds_from_moist_block_type), intent(in)  :: Moist_clouds_block
    ! swdn_toa is input
    real, dimension(:,:,:),       intent(inout)     :: tdt_sw, tdt_lw, tdt_sw_clr, tdt_lw_clr
    real, dimension(:,:),         intent(inout)     :: lwdn_sfc, lwup_sfc, swdn_sfc, swup_sfc,  &
                                                       swdn_toa, swup_toa, olr, & 
                                                       lwdn_sfc_clr, swdn_sfc_clr, swup_sfc_clr, &
                                                       swup_toa_clr, olr_clr
    !---------------------------------------------------------------------
    ! local variables
    integer :: n, kmax
    logical :: flag            

    !3D heating rate
    if (id_nn_tdt_lw     .gt. 0) flag = send_data(id_nn_tdt_lw    , tdt_lw    , time_next, is, js, 1)
    if (id_nn_tdt_sw     .gt. 0) flag = send_data(id_nn_tdt_sw    , tdt_sw    , time_next, is, js, 1)
    if (id_nn_tdt_lw_clr .gt. 0) flag = send_data(id_nn_tdt_lw_clr, tdt_lw_clr, time_next, is, js, 1)
    if (id_nn_tdt_sw_clr .gt. 0) flag = send_data(id_nn_tdt_sw_clr, tdt_sw_clr, time_next, is, js, 1)
    
    !2D boundary flux
    if (id_nn_lwdn_sfc .gt. 0) flag = send_data(id_nn_lwdn_sfc, lwdn_sfc, time_next, is, js)
    if (id_nn_lwup_sfc .gt. 0) flag = send_data(id_nn_lwup_sfc, lwup_sfc, time_next, is, js)
    if (id_nn_swdn_sfc .gt. 0) flag = send_data(id_nn_swdn_sfc, swdn_sfc, time_next, is, js)
    if (id_nn_swup_sfc .gt. 0) flag = send_data(id_nn_swup_sfc, swup_sfc, time_next, is, js)
    if (id_nn_swdn_toa .gt. 0) flag = send_data(id_nn_swdn_toa, swdn_toa, time_next, is, js)
    if (id_nn_swup_toa .gt. 0) flag = send_data(id_nn_swup_toa, swup_toa, time_next, is, js)
    if (id_nn_olr      .gt. 0) flag = send_data(id_nn_olr     , olr     , time_next, is, js)
    
    if (id_nn_olr_clr      .gt. 0) flag = send_data(id_nn_olr_clr     , olr_clr     , time_next, is, js)
    if (id_nn_lwdn_sfc_clr .gt. 0) flag = send_data(id_nn_lwdn_sfc_clr, lwdn_sfc_clr, time_next, is, js)
    if (id_nn_swdn_sfc_clr .gt. 0) flag = send_data(id_nn_swdn_sfc_clr, swdn_sfc_clr, time_next, is, js)
    if (id_nn_swup_sfc_clr .gt. 0) flag = send_data(id_nn_swup_sfc_clr, swup_sfc_clr, time_next, is, js)
    if (id_nn_swup_toa_clr .gt. 0) flag = send_data(id_nn_swup_toa_clr, swup_toa_clr, time_next, is, js)
    ! send_data to diag files
    !3D atmosphere fields.
    if (idtlev .gt. 0) flag = send_data(idtlev, tflux, time_next, is, js, 1)
    if (idtlay .gt. 0) flag = send_data(idtlay, temp,  time_next, is, js, 1)
    if (idh2o  .gt. 0) flag = send_data(idh2o,  rh2o,  time_next, is, js, 1)
    if (ido3   .gt. 0) flag = send_data(ido3,   rad_gases%qo3,     time_next, is, js, 1)

    !Clouds
    n = moist_clouds_block%index_strat
    if (idsd .gt. 0) flag = send_data(idsd, moist_clouds_block%cloud_data(n)%droplet_number, &
                                      time_next, is, js, 1)
    if (idsf .gt. 0) flag = send_data(idsf, moist_clouds_block%cloud_data(n)%cloud_area, &
                                      time_next, is, js, 1)
    if (idsl .gt. 0) flag = send_data(idsl, moist_clouds_block%cloud_data(n)%liquid_amt, &
                                      time_next, is, js, 1)
    if (idsi .gt. 0) flag = send_data(idsi, moist_clouds_block%cloud_data(n)%ice_amt, &
                                      time_next, is, js, 1) 
    n = moist_clouds_block%index_uw_conv
    if (idcd .gt. 0) flag = send_data(idcd, moist_clouds_block%cloud_data(n)%droplet_number, &
                                      time_next, is, js, 1)
    if (idcf .gt. 0) flag = send_data(idcf, moist_clouds_block%cloud_data(n)%cloud_area, &
                                      time_next, is, js, 1)
    if (idcl .gt. 0) flag = send_data(idcl, moist_clouds_block%cloud_data(n)%liquid_amt, &
                                      time_next, is, js, 1)
    if (idci .gt. 0) flag = send_data(idci, moist_clouds_block%cloud_data(n)%ice_amt, &
                                      time_next, is, js, 1)
    !2D atmosphere fields.
    kmax = size(phalf,3)
    if (idts   .gt. 0) flag = send_data(idts,   tsfc,         time_next, is, js)
    if (idps   .gt. 0) flag = send_data(idps,   phalf(:,:,kmax),    time_next, is, js)
    if (idvdir .gt. 0) flag = send_data(idvdir, asfc_vis_dir, time_next, is, js)
    if (idvdif .gt. 0) flag = send_data(idvdif, asfc_vis_dif, time_next, is, js)
    if (ididir .gt. 0) flag = send_data(ididir, asfc_nir_dir, time_next, is, js)
    if (ididif .gt. 0) flag = send_data(ididif, asfc_nir_dif, time_next, is, js)

    if (idsolarfact  .gt. 0) flag = send_data(idsolarfact, Astro%solar  , time_next, is, js)
    if (idzen        .gt. 0) flag = send_data(idzen      , Astro%cosz   , time_next, is, js)
    if (idfrac       .gt. 0) flag = send_data(idfrac     , Astro%fracday, time_next, is, js)
    
    !1D
    if (idsolar .gt. 0) flag = send_data(idsolar, solar_constant_used,  time_next)
    if (ider    .gt. 0) flag = send_data(ider   , Astro%rrsun ,         time_next)
end subroutine produce_rad_nn_diag

subroutine test ()
    call error_mesg('radiation_driver_nn_mod', 'call test', NOTE)

end subroutine test


!#######################################################################

end module radiation_driver_nn_mod
