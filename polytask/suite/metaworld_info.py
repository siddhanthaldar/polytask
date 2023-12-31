import  metaworld.policies as policies

env_names = [
	'button-press-topdown-v2',
	'button-press-topdown-wall-v2',
	'button-press-v2',
	'button-press-wall-v2',
	'coffee-button-v2',
	'coffee-pull-v2',
	'door-open-v2',
	'door-unlock-v2',
	'drawer-close-v2',
	'drawer-open-v2',
	'hammer-v2',
	'plate-slide-v2',
	'plate-slide-side-v2',
	'plate-slide-back-v2',
	'plate-slide-back-side-v2',
	'bin-picking-v2',
]

POLICY = {
	'button-press-topdown-v2': policies.SawyerButtonPressTopdownV2Policy,
	'button-press-topdown-wall-v2': policies.SawyerButtonPressTopdownWallV2Policy,
	'button-press-v2': policies.SawyerButtonPressV2Policy,
	'button-press-wall-v2': policies.SawyerButtonPressWallV2Policy,
	'coffee-button-v2': policies.SawyerCoffeeButtonV2Policy,
	'coffee-pull-v2': policies.SawyerCoffeePullV2Policy,
	'door-open-v2': policies.SawyerDoorOpenV2Policy,
	'door-unlock-v2': policies.SawyerDoorUnlockV2Policy,
	'drawer-close-v2': policies.SawyerDrawerCloseV2Policy,
	'drawer-open-v2': policies.SawyerDrawerOpenV2Policy,
	'hammer-v2': policies.SawyerHammerV2Policy,
	'plate-slide-v2': policies.SawyerPlateSlideV2Policy,
	'plate-slide-side-v2': policies.SawyerPlateSlideSideV2Policy,
	'plate-slide-back-v2': policies.SawyerPlateSlideBackV2Policy,
	'plate-slide-back-side-v2': policies.SawyerPlateSlideBackSideV2Policy,
	'bin-picking-v2': policies.SawyerBinPickingV2Policy,
}

CAMERA = {
	'button-press-topdown-v2': 'corner',
	'button-press-topdown-wall-v2': 'corner',
	'button-press-v2': 'corner',
	'button-press-wall-v2': 'corner',
	'coffee-button-v2': 'corner',
	'coffee-pull-v2': 'corner',
	'door-open-v2': 'corner3',
	'door-unlock-v2': 'corner',
	'drawer-close-v2': 'corner',
	'drawer-open-v2': 'corner',
	'hammer-v2': 'corner3',
	'plate-slide-v2': 'corner3',
	'plate-slide-side-v2': 'corner3',
	'plate-slide-back-v2': 'corner3',
	'plate-slide-back-side-v2': 'corner3',
	'bin-picking-v2': 'corner',
}

MAX_PATH_LENGTH = {
	'button-press-topdown-v2': 125,
	'button-press-topdown-wall-v2': 125,
	'button-press-v2': 125,
	'button-press-wall-v2': 125,
	'coffee-button-v2': 125,
    'coffee-pull-v2': 125,
    'door-open-v2': 125,
	'door-unlock-v2': 125,
	'drawer-close-v2': 125,
	'drawer-open-v2': 125,
	'hammer-v2': 125,
	'plate-slide-v2': 125,
	'plate-slide-side-v2': 125,
	'plate-slide-back-v2': 125,
	'plate-slide-back-side-v2': 125,
	'bin-picking-v2': 175,
}
