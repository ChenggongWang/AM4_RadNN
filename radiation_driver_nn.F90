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
! default private
private 
! cgw: for NN_diag
integer :: idtlev, idplay, idtlay, idh2o, ido3, idzlay, idsd, &
           idsf, idsl, idsi, idcd, idcf, idcl, idci, &
           idph, idts, idzen, &
           idvdir, idvdif, ididir, ididif, idfrac, idland, ider,idsolarfact, idsolar, &
           id_nn_tdt_lw, id_nn_tdt_sw, &
           id_nn_lwdn_sfc, id_nn_lwup_sfc, id_nn_swdn_sfc, id_nn_swup_sfc, &
           id_nn_swdn_toa, id_nn_swup_toa, id_nn_olr, &
           id_nn_tdt_lw_clr,id_nn_tdt_sw_clr, &
           id_nn_lwdn_sfc_clr, id_nn_swdn_sfc_clr, id_nn_swup_sfc_clr, &
           id_nn_swup_toa_clr, id_nn_olr_clr 
character(len=32) :: mod_name="NN_diag"

!----------------------------------------------------------------------
! cgw
! define type for linear layer
! contains weight(2d) and bias(1d)
!----------------------------------------------------------------------
private NN_Linear_layer_type
type :: NN_Linear_layer_type 
    real(kind=4), dimension(:,:), pointer :: weight=>NULL()
    real(kind=4), dimension(:),   pointer :: bias=>NULL()
end type NN_Linear_layer_type
private NN_FC_type
type :: NN_FC_type
    integer :: num_hid_nodes
    integer :: num_layers
    type(NN_Linear_layer_type), dimension(:), pointer:: Layers
end type NN_FC_type

! cgw: for NN_para
type(NN_FC_type), dimension(4):: Rad_NN_FC

! public subroutines used by radiation_driver.F90
public  radiation_driver_nn_init, &
        NN_radiation_calc, &
        produce_rad_nn_diag


                         contains

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
!                     PUBLIC SUBROUTINES
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subroutine radiation_driver_nn_init(do_rad_nn, axes, time, rad_nn_para_nc)
    logical,                    intent(in)  :: do_rad_nn
    type(time_type),            intent(in)  :: time
    integer, dimension(4),      intent(in)  :: axes
    character(len=32),          intent(in)  :: rad_nn_para_nc
    
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
        id_nn_olr      = register_diag_field(mod_name, "nn_olr",      a(1:2), time,  "nn_olr", "Wm-2")
        id_nn_lwdn_sfc_clr = register_diag_field(mod_name, "nn_lwdn_sfc_clr", a(1:2), time,  "nn_lwdn_sfc_clr", "Wm-2")
        id_nn_swdn_sfc_clr = register_diag_field(mod_name, "nn_swdn_sfc_clr", a(1:2), time,  "nn_swdn_sfc_clr", "Wm-2")
        id_nn_swup_sfc_clr = register_diag_field(mod_name, "nn_swup_sfc_clr", a(1:2), time,  "nn_swup_sfc_clr", "Wm-2")
        id_nn_swup_toa_clr = register_diag_field(mod_name, "nn_swup_toa_clr", a(1:2), time,  "nn_swup_toa_clr", "Wm-2")
        id_nn_olr_clr      = register_diag_field(mod_name, "nn_olr_clr",      a(1:2), time,  "nn_olr_clr", "Wm-2")
    
    end if ! do_rad_nn
    ! diag
    !write(outunit,*) 'register output mod: ', mod_name
    a(1:2) = axes(1:2)
    a(3) = axes(4)
    !3D atmosphere fields.
    ! phalf k+1 levels
    idtlev = register_diag_field(mod_name, "tflux", a(1:3), time, &
                                 "level temperature", "K")
    idph   = register_diag_field(mod_name, "plevel", a(1:3), time, &
                                 "level pressure", "Pa")
    
    ! pfull k levels
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
                                 "solar factor", "none")
    idzen = register_diag_field(mod_name, "cosz", a(1:2), time, &
                                "cosine solar zenith angle", "none")
    idfrac = register_diag_field(mod_name, "dayf", a(1:2), time, &
                               "daylight fraction", "none")
    !1D 
    idsolar = register_diag_field(mod_name, "solar_constant", time, &
                                  "solar constant", "none")
    ! earth_sun_distance_fraction
    ider = register_diag_field(mod_name, "rrsun", time, &
                               "rrsun factor", "none")

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
    integer :: n
    logical :: flag            

    ! send_data to diag files: output of NN
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

    ! send_data to diag files: input for NN
    !3D atmosphere fields.
    if (idph   .gt. 0) flag = send_data(idph,   phalf, time_next, is, js, 1)
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
    if (idts   .gt. 0) flag = send_data(idts,   tsfc,         time_next, is, js)
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

!######################################################################
! cgw: function and subroutine for an NN to predict
! NN activation function
real elemental function NN_activ(x)
    real(kind=4), intent(in) :: x
    ! ReLU:
    NN_activ = max(0.0,x)
    ! tanh
    ! y = tanh(x)
end function NN_activ
subroutine nn_pred_1d_sgemm(FNN,x,y)
    type(NN_FC_type),   intent(in)    :: FNN
    real(kind=4), dimension(:), intent(in)    :: x
    real(kind=4), dimension(:), intent(inout) :: y
    ! local
    integer :: ilayer
    real(kind=4), dimension(:), allocatable :: interm1, interm2
    ! for sgemm
    integer :: m, k, n
    real(kind=4) :: alpha, beta
    alpha = 1.0
    beta = 1.0

    allocate(interm1(size(x)))
    interm1 = x
    do ilayer = 1, FNN%num_layers
        m = 1
        k = size(interm1)
        n = size(FNN%Layers(ilayer)%bias)
        allocate(interm2(n))
        interm2 = FNN%Layers(ilayer)%bias
        !call SGEMM('N','N',m,n,k,1.0,interm1,m,FNN%Layers(ilayer)%weight,k,1.0,interm2,m)
        call SGEMV('T',k,n,alpha,FNN%Layers(ilayer)%weight,k,interm1,1,beta,interm2,1)
        interm2 = NN_activ(interm2)
        deallocate(interm1)
        allocate(interm1(n))
        interm1 = interm2
        deallocate(interm2)
    end do
    y = interm1
    deallocate(interm1)
end subroutine nn_pred_1d_sgemm

subroutine nn_pred_1d_matmul(FNN,x,y)
    type(NN_FC_type),   intent(in) :: FNN
    real(kind=4), dimension(:), intent(in) :: x
    real(kind=4), dimension(:), intent(inout) :: y
    real(kind=4), dimension(:), allocatable :: interm1, interm2
    integer :: ilayer, n
    allocate(interm1(size(x)))
    interm1 = x
    ! num_layers matmul called
    do ilayer = 1, FNN%num_layers
        n = size(FNN%Layers(ilayer)%bias)
        allocate(interm2(n))
        interm2 = matmul(interm1,FNN%Layers(ilayer)%weight) + FNN%Layers(ilayer)%bias
        interm2 = NN_activ(interm2)
        deallocate(interm1)
        allocate(interm1(n))
        interm1 = interm2
        deallocate(interm2)
    end do
    y = interm1
    deallocate(interm1)
end subroutine nn_pred_1d_matmul
!######################################################################
! cgw: subroutine to apply NN 
!      only consider ozone and cloud, no aerosol and other GHGs for now
!
!
subroutine NN_radiation_calc (phalf, temp, tflux,  tsfc, rh2o, Rad_gases, Astro, solar_constant_used, &
                              asfc_vis_dir, asfc_vis_dif, asfc_nir_dir, asfc_nir_dif, &
                              moist_clouds_block, &
                              tdt_sw, tdt_lw, &
                              lwdn_sfc, lwup_sfc, swdn_sfc, swup_sfc,  &
                              swdn_toa, swup_toa, olr, & 
                              tdt_sw_clr, tdt_lw_clr, &
                              lwdn_sfc_clr, swdn_sfc_clr, swup_sfc_clr, &
                              swup_toa_clr, olr_clr ) 


    !--------------------------------------------------------------------
    real, dimension(:,:,:),       intent(in)             :: phalf, temp, &
                                                            tflux, rh2o
    real, dimension(:,:),         intent(in)             :: tsfc
    type(radiative_gases_type),   intent(in)             :: Rad_gases
    type(astronomy_type),         intent(in)             :: Astro
    real,                         intent(in)             :: solar_constant_used
    real, dimension(:,:),         intent(in)             :: asfc_vis_dir, &
                                                            asfc_nir_dir, &
                                                            asfc_vis_dif, &
                                                            asfc_nir_dif
    type(clouds_from_moist_block_type), intent(in)       :: Moist_clouds_block
    ! swdn_toa is input
    real, dimension(:,:,:),       intent(inout)          :: tdt_sw, tdt_lw, tdt_sw_clr, tdt_lw_clr
    real, dimension(:,:),         intent(inout)          :: lwdn_sfc, lwup_sfc, swdn_sfc, swup_sfc,  &
                                                            swdn_toa, swup_toa, olr, & 
                                                            lwdn_sfc_clr, swdn_sfc_clr, swup_sfc_clr, &
                                                            swup_toa_clr, olr_clr
    !---------------------------------------------------------------------
    ! local variables
    integer :: i, j, isize, jsize, ksize, outunit, inn, cstra, cconv
    real(kind=4), allocatable, dimension(:) :: input_X, output_Y
    isize = size(temp,1)
    jsize = size(temp,2)
    ksize = size(temp,3)
    cstra = Moist_clouds_block%index_strat
    cconv = Moist_clouds_block%index_uw_conv
    ! loop over all locations, might be faster if do in all location
    ! need to optimize/test in next dev
    ! v0: for lwcs, input_X(102) , this will be change in the future version
    inn = 1 
    allocate(input_X(size(Rad_NN_FC(inn)%layers(1)%weight,1)))
    allocate(output_y(size(Rad_NN_FC(inn)%layers(Rad_NN_FC(inn)%num_layers)%bias)))
    do j = 1, jsize
        do i = 1, isize
            input_X(1) = phalf(i,j,ksize+1)   ! ps
            input_X(2:2+ksize) = tflux(i,j,:) ! need to update to temp, since tflux is from tsfc
            input_X(3+ksize)   = tsfc(i,j)
            input_X(4+ksize:3+2*ksize) = rh2o(i,j,:)
            input_X(4+2*ksize:3+3*ksize) = Rad_gases%qo3(i,j,:)
            call NN_pred_1d_sgemm (Rad_NN_FC(inn), input_X, output_Y)
            lwdn_sfc_clr(i,j) = output_Y(1) 
            lwup_sfc(i,j)     = output_Y(2) 
            olr_clr(i,j)      = output_Y(3) 
            tdt_lw_clr(i,j,:) = output_Y(4:) 
        end do
    end do
    deallocate(input_X, output_Y)
    ! v0.1: for lw, input_X(3xx) , this will be change in the future version
    inn = 2
    allocate(input_X(size(Rad_NN_FC(inn)%layers(1)%weight,1)))
    allocate(output_y(size(Rad_NN_FC(inn)%layers(Rad_NN_FC(inn)%num_layers)%bias)))
    do j = 1, jsize
        do i = 1, isize
            input_X(1) = phalf(i,j,ksize+1)   ! ps
            input_X(2:2+ksize) = tflux(i,j,:) ! need to update to temp, since tflux is from tsfc
            input_X(3+ksize) = tsfc(i,j)
            input_X(4+ksize:3+2*ksize) = rh2o(i,j,:)
            input_X(4+ 2*ksize:3+ 3*ksize) = Rad_gases%qo3(i,j,:)
            input_X(4+ 3*ksize:3+ 4*ksize) = Moist_clouds_block%cloud_data(cstra)%droplet_number(i,j,:)
            input_X(4+ 4*ksize:3+ 5*ksize) = Moist_clouds_block%cloud_data(cstra)%cloud_area(i,j,:)
            input_X(4+ 5*ksize:3+ 6*ksize) = Moist_clouds_block%cloud_data(cstra)%liquid_amt(i,j,:)
            input_X(4+ 6*ksize:3+ 7*ksize) = Moist_clouds_block%cloud_data(cstra)%ice_amt(i,j,:)
            input_X(4+ 7*ksize:3+ 8*ksize) = Moist_clouds_block%cloud_data(cconv)%droplet_number(i,j,:)
            input_X(4+ 8*ksize:3+ 9*ksize) = Moist_clouds_block%cloud_data(cconv)%cloud_area(i,j,:)
            input_X(4+ 9*ksize:3+10*ksize) = Moist_clouds_block%cloud_data(cconv)%liquid_amt(i,j,:)
            input_X(4+10*ksize:3+11*ksize) = Moist_clouds_block%cloud_data(cconv)%ice_amt(i,j,:)
            call NN_pred_1d_sgemm (Rad_NN_FC(inn), input_X, output_Y)
            lwdn_sfc(i,j) = output_Y(1) 
            lwup_sfc(i,j)     = output_Y(2) 
            olr(i,j)      = output_Y(3) 
            tdt_lw(i,j,:) = output_Y(4:) 
        end do
    end do
    deallocate(input_X, output_Y)
    ! v0.2: for swcs, input_X(1xx) , this will be change in the future version
    inn = 3
    allocate(input_X(size(Rad_NN_FC(inn)%layers(1)%weight,1)))
    allocate(output_y(size(Rad_NN_FC(inn)%layers(Rad_NN_FC(inn)%num_layers)%bias)))
    do j = 1, jsize
        do i = 1, isize
            input_X(1) = phalf(i,j,ksize+1)   ! ps
            input_X(2) = solar_constant_used*Astro%solar(i,j)   ! rsdt
            swdn_toa(i,j) = input_X(2)
            if (swdn_toa(i,j) > 1e-3) then !daylight
                input_X(3:3+ksize) = tflux(i,j,:) ! need to update to temp, since tflux is from tsfc
                input_X(4+ksize) = tsfc(i,j)
                input_X(5+ksize:4+2*ksize) = rh2o(i,j,:)
                input_X(5+2*ksize:4+3*ksize) = Rad_gases%qo3(i,j,:)
                input_X(5+3*ksize) = Astro%cosz(i,j)
                input_X(6+3*ksize) = asfc_vis_dir(i,j)
                input_X(7+3*ksize) = asfc_vis_dif(i,j)
                input_X(8+3*ksize) = asfc_nir_dir(i,j)
                input_X(9+3*ksize) = asfc_nir_dif(i,j)
                call NN_pred_1d_sgemm (Rad_NN_FC(inn), input_X, output_Y)
                swup_toa_clr(i,j) = output_Y(1) 
                swdn_sfc_clr(i,j) = output_Y(2) 
                swup_sfc_clr(i,j) = output_Y(3) 
                tdt_sw_clr(i,j,:) = output_Y(4:) 
            else
                swup_toa_clr(i,j) = 0.0
                swdn_sfc_clr(i,j) = 0.0
                swup_sfc_clr(i,j) = 0.0
                tdt_sw_clr(i,j,:) = 0.0 
            end if
        end do
    end do
    deallocate(input_X, output_Y)
    ! v0.3: for sw, input_X(3xx) , this will be change in the future version
    inn = 4
    allocate(input_X(size(Rad_NN_FC(inn)%layers(1)%weight,1)))
    allocate(output_y(size(Rad_NN_FC(inn)%layers(Rad_NN_FC(inn)%num_layers)%bias)))
    do j = 1, jsize
        do i = 1, isize
            input_X(1) = phalf(i,j,ksize+1)   ! ps
            input_X(2) = swdn_toa(i,j)        ! rsdt
            swdn_toa(i,j) = input_X(2)
            if (swdn_toa(i,j) > 1e-3) then !daylight
                input_X(3:3+ksize) = tflux(i,j,:) ! need to update to temp, since tflux is from tsfc
                input_X(4+ksize) = tsfc(i,j)
                input_X(5+ksize:4+2*ksize) = rh2o(i,j,:)
                input_X(5+2*ksize:4+3*ksize) = Rad_gases%qo3(i,j,:)
                input_X(5+3*ksize) = Astro%cosz(i,j)
                input_X(6+3*ksize) = asfc_vis_dir(i,j)
                input_X(7+3*ksize) = asfc_vis_dif(i,j)
                input_X(8+3*ksize) = asfc_nir_dir(i,j)
                input_X(9+3*ksize) = asfc_nir_dif(i,j)
                input_X(10+ 3*ksize:9+ 4*ksize) = Moist_clouds_block%cloud_data(cstra)%droplet_number(i,j,:)
                input_X(10+ 4*ksize:9+ 5*ksize) = Moist_clouds_block%cloud_data(cstra)%cloud_area(i,j,:)
                input_X(10+ 5*ksize:9+ 6*ksize) = Moist_clouds_block%cloud_data(cstra)%liquid_amt(i,j,:)
                input_X(10+ 6*ksize:9+ 7*ksize) = Moist_clouds_block%cloud_data(cstra)%ice_amt(i,j,:)
                input_X(10+ 7*ksize:9+ 8*ksize) = Moist_clouds_block%cloud_data(cconv)%droplet_number(i,j,:)
                input_X(10+ 8*ksize:9+ 9*ksize) = Moist_clouds_block%cloud_data(cconv)%cloud_area(i,j,:)
                input_X(10+ 9*ksize:9+10*ksize) = Moist_clouds_block%cloud_data(cconv)%liquid_amt(i,j,:)
                input_X(10+10*ksize:9+11*ksize) = Moist_clouds_block%cloud_data(cconv)%ice_amt(i,j,:)
                call NN_pred_1d_sgemm (Rad_NN_FC(inn), input_X, output_Y)
                swup_toa(i,j) = output_Y(1) 
                swdn_sfc(i,j) = output_Y(2) 
                swup_sfc(i,j) = output_Y(3) 
                tdt_sw(i,j,:) = output_Y(4:) 
            else
                swup_toa(i,j) = 0.0
                swdn_sfc(i,j) = 0.0
                swup_sfc(i,j) = 0.0
                tdt_sw(i,j,:) = 0.0 
            end if
        end do
    end do
    deallocate(input_X, output_Y)
    
end subroutine NN_radiation_calc


!#######################################################################

end module radiation_driver_nn_mod
