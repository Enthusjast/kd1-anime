"""Generated metadata from trusted Manim; contains no upstream source code."""

CATALOG = {
    "schema_version": 1,
    "symbols": {
        "Add": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.animation",
            "signature": "(*mobjects: 'Mobject', run_time: 'float' = 0.0, **kwargs: "
            "'Any') -> 'None'",
        },
        "AddTextLetterByLetter": {
            "bases": ["ShowIncreasingSubsets", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.creation",
            "signature": "(text: 'Text', suspend_mobject_updating: "
            "'bool' = False, int_func: "
            "'Callable[[np.ndarray], np.ndarray]' = <ufunc "
            "'ceil'>, rate_func: 'Callable[[float], "
            "float]' = <function linear>, time_per_char: "
            "'float' = 0.1, run_time: 'float | None' = "
            "None, reverse_rate_function=False, "
            "introducer=True, **kwargs) -> 'None'",
        },
        "AddTextWordByWord": {
            "bases": ["Succession", "AnimationGroup", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.creation",
            "signature": "(text_mobject: 'Text', run_time: 'float' = None, "
            "time_per_char: 'float' = 0.06, **kwargs) -> "
            "'None'",
        },
        "Angle": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.line",
            "signature": "(line1: 'Line', line2: 'Line', radius: 'float | None' = None, "
            "quadrant: 'AngleQuadrant' = (1, 1), other_angle: 'bool' = "
            "False, dot: 'bool' = False, dot_radius: 'float | None' = "
            "None, dot_distance: 'float' = 0.55, dot_color: "
            "'ParsableManimColor' = ManimColor('#FFFFFF'), elbow: 'bool' = "
            "False, **kwargs: 'Any') -> 'None'",
        },
        "AnimatedBoundary": {
            "bases": ["VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.animation.changing",
            "signature": "(vmobject: 'VMobject', colors: "
            "'Sequence[ParsableManimColor]' = "
            "[ManimColor('#29ABCA'), ManimColor('#9CDCEB'), "
            "ManimColor('#236B8E'), ManimColor('#736357')], "
            "max_stroke_width: 'float' = 3, cycle_rate: 'float' "
            "= 0.5, back_and_forth: 'bool' = True, "
            "draw_rate_func: 'RateFunction' = <function "
            "smooth>, fade_rate_func: 'RateFunction' = "
            "<function smooth>, **kwargs: 'Any')",
        },
        "Animation": {
            "bases": ["object"],
            "kind": "class",
            "module": "manim.animation.animation",
            "signature": "(mobject=None, *args, use_override=True, **kwargs) -> 'Self'",
        },
        "AnimationGroup": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.composition",
            "signature": "(*animations: 'Animation | Iterable[Animation]', "
            "group: 'Group | VGroup | OpenGLGroup | OpenGLVGroup "
            "| None' = None, run_time: 'float | None' = None, "
            "rate_func: 'Callable[[float], float]' = <function "
            "linear>, lag_ratio: 'float' = 0, **kwargs: 'Any')",
        },
        "AnnotationDot": {
            "bases": ["Dot", "Circle", "Arc", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(radius: 'float' = 0.10400000000000001, stroke_width: "
            "'float' = 5, stroke_color: 'ParsableManimColor' = "
            "ManimColor('#FFFFFF'), fill_color: "
            "'ParsableManimColor' = ManimColor('#58C4DD'), "
            "**kwargs: 'Any') -> 'None'",
        },
        "AnnularSector": {
            "bases": ["Arc", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(inner_radius: 'float' = 1, outer_radius: 'float' = "
            "2, angle: 'float' = 1.5707963267948966, start_angle: "
            "'float' = 0, fill_opacity: 'float' = 1, stroke_width: "
            "'float' = 0, color: 'ParsableManimColor' = "
            "ManimColor('#FFFFFF'), **kwargs: 'Any') -> 'None'",
        },
        "Annulus": {
            "bases": ["Circle", "Arc", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(inner_radius: 'float' = 1, outer_radius: 'float' = 2, "
            "fill_opacity: 'float' = 1, stroke_width: 'float' = 0, "
            "color: 'ParsableManimColor' = ManimColor('#FFFFFF'), "
            "mark_paths_closed: 'bool' = False, **kwargs: 'Any') -> "
            "'None'",
        },
        "ApplyComplexFunction": {
            "bases": ["ApplyMethod", "Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(function: 'types.MethodType', mobject: 'Mobject', **kwargs) -> 'None'",
        },
        "ApplyFunction": {
            "bases": ["Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(function: 'types.MethodType', mobject: 'Mobject', **kwargs) -> 'None'",
        },
        "ApplyMatrix": {
            "bases": ["ApplyPointwiseFunction", "ApplyMethod", "Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(matrix: 'np.ndarray', mobject: 'Mobject', about_point: "
            "'np.ndarray' = array([0., 0., 0.]), **kwargs) -> "
            "'None'",
        },
        "ApplyMethod": {
            "bases": ["Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(method: 'Callable', *args, **kwargs) -> 'None'",
        },
        "ApplyPointwiseFunction": {
            "bases": ["ApplyMethod", "Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(function: 'types.MethodType', mobject: "
            "'Mobject', run_time: 'float' = 3.0, "
            "**kwargs) -> 'None'",
        },
        "ApplyPointwiseFunctionToCenter": {
            "bases": ["ApplyPointwiseFunction", "ApplyMethod", "Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(function: 'types.MethodType', mobject: 'Mobject', **kwargs) -> 'None'",
        },
        "ApplyWave": {
            "bases": ["Homotopy", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.indication",
            "signature": "(mobject: 'Mobject', direction: 'Vector3DLike' = "
            "array([0., 1., 0.]), amplitude: 'float' = 0.2, wave_func: "
            "'RateFunction' = <function smooth>, time_width: 'float' = "
            "1, ripples: 'int' = 1, run_time: 'float' = 2, **kwargs: "
            "'Any')",
        },
        "Arc": {
            "bases": ["TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(radius: 'float | None' = 1.0, start_angle: 'float' = 0, angle: "
            "'float' = 1.5707963267948966, num_components: 'int' = 9, "
            "arc_center: 'Point3DLike' = array([0., 0., 0.]), **kwargs: "
            "'Any')",
        },
        "ArcBetweenPoints": {
            "bases": ["Arc", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(start: 'Point3DLike', end: 'Point3DLike', angle: "
            "'float' = 1.5707963267948966, radius: 'float | "
            "None' = None, **kwargs: 'Any') -> 'None'",
        },
        "ArcBrace": {
            "bases": ["Brace", "VMobjectFromSVGPath", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.svg.brace",
            "signature": "(arc: 'Arc | None' = None, direction: 'Vector3DLike' = "
            "array([1., 0., 0.]), **kwargs: 'Any')",
        },
        "ArcPolygon": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(*vertices: 'Point3DLike', angle: 'float' = "
            "0.7853981633974483, radius: 'float | None' = None, "
            "arc_config: 'list[dict] | None' = None, **kwargs: 'Any') "
            "-> 'None'",
        },
        "ArcPolygonFromArcs": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(*arcs: 'Arc | ArcBetweenPoints', **kwargs: 'Any') -> 'None'",
        },
        "Arrow": {
            "bases": ["Line", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.line",
            "signature": "(*args: 'Any', stroke_width: 'float' = 6, buff: 'float' = "
            "0.25, max_tip_length_to_length_ratio: 'float' = 0.25, "
            "max_stroke_width_to_length_ratio: 'float' = 5, **kwargs: "
            "'Any') -> 'None'",
        },
        "Arrow3D": {
            "bases": ["Line3D", "Cylinder", "Surface", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.three_dimensions",
            "signature": "(start: 'Point3DLike' = array([-1.,  0.,  0.]), end: "
            "'Point3DLike' = array([1., 0., 0.]), thickness: 'float' = "
            "0.02, height: 'float' = 0.3, base_radius: 'float' = 0.08, "
            "color: 'ParsableManimColor' = ManimColor('#FFFFFF'), "
            "resolution: 'int | tuple[int, int]' = 24, **kwargs: 'Any') "
            "-> 'None'",
        },
        "ArrowCircleFilledTip": {
            "bases": [
                "ArrowCircleTip",
                "ArrowTip",
                "Circle",
                "Arc",
                "TipableVMobject",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.geometry.tips",
            "signature": "(fill_opacity: 'float' = 1, stroke_width: "
            "'float' = 0, **kwargs: 'Any') -> 'None'",
        },
        "ArrowCircleTip": {
            "bases": [
                "ArrowTip",
                "Circle",
                "Arc",
                "TipableVMobject",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.geometry.tips",
            "signature": "(fill_opacity: 'float' = 0, stroke_width: 'float' = "
            "3, length: 'float' = 0.35, start_angle: 'float' = "
            "3.141592653589793, **kwargs: 'Any') -> 'None'",
        },
        "ArrowSquareFilledTip": {
            "bases": [
                "ArrowSquareTip",
                "ArrowTip",
                "Square",
                "Rectangle",
                "Polygon",
                "Polygram",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.geometry.tips",
            "signature": "(fill_opacity: 'float' = 1, stroke_width: "
            "'float' = 0, **kwargs: 'Any') -> 'None'",
        },
        "ArrowSquareTip": {
            "bases": [
                "ArrowTip",
                "Square",
                "Rectangle",
                "Polygon",
                "Polygram",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.geometry.tips",
            "signature": "(fill_opacity: 'float' = 0, stroke_width: 'float' = "
            "3, length: 'float' = 0.35, start_angle: 'float' = "
            "3.141592653589793, **kwargs: 'Any') -> 'None'",
        },
        "ArrowTip": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.tips",
            "signature": "(*args: 'Any', **kwargs: 'Any') -> 'None'",
        },
        "ArrowTriangleFilledTip": {
            "bases": [
                "ArrowTriangleTip",
                "ArrowTip",
                "Triangle",
                "RegularPolygon",
                "RegularPolygram",
                "Polygram",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.geometry.tips",
            "signature": "(fill_opacity: 'float' = 1, stroke_width: "
            "'float' = 0, **kwargs: 'Any') -> 'None'",
        },
        "ArrowTriangleTip": {
            "bases": [
                "ArrowTip",
                "Triangle",
                "RegularPolygon",
                "RegularPolygram",
                "Polygram",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.geometry.tips",
            "signature": "(fill_opacity: 'float' = 0, stroke_width: 'float' "
            "= 3, length: 'float' = 0.35, width: 'float' = "
            "0.35, start_angle: 'float' = 3.141592653589793, "
            "**kwargs: 'Any') -> 'None'",
        },
        "ArrowVectorField": {
            "bases": ["VectorField", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.vector_field",
            "signature": "(func: 'Callable[[np.ndarray], np.ndarray]', "
            "color: 'ParsableManimColor | None' = None, "
            "color_scheme: 'Callable[[np.ndarray], float] | "
            "None' = None, min_color_scheme_value: 'float' = 0, "
            "max_color_scheme_value: 'float' = 2, colors: "
            "'Sequence[ParsableManimColor]' = "
            "[ManimColor('#236B8E'), ManimColor('#83C167'), "
            "ManimColor('#F7D96F'), ManimColor('#FC6255')], "
            "x_range: 'Sequence[float]' = None, y_range: "
            "'Sequence[float]' = None, z_range: "
            "'Sequence[float]' = None, three_dimensions: 'bool' "
            "= False, length_func: 'Callable[[float], float]' = "
            "<function ArrowVectorField.<lambda>>, opacity: "
            "'float' = 1.0, vector_config: 'dict | None' = "
            "None, **kwargs)",
        },
        "Axes": {
            "bases": ["VGroup", "VMobject", "Mobject", "CoordinateSystem", "object"],
            "kind": "class",
            "module": "manim.mobject.graphing.coordinate_systems",
            "signature": "(x_range: 'Sequence[float] | None' = None, y_range: "
            "'Sequence[float] | None' = None, x_length: 'float | None' = "
            "12, y_length: 'float | None' = 6, axis_config: 'dict | None' = "
            "None, x_axis_config: 'dict | None' = None, y_axis_config: "
            "'dict | None' = None, tips: 'bool' = True, **kwargs: 'Any')",
        },
        "BLACK": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "BLUE": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "BLUE_A": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "BLUE_B": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "BLUE_C": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "BLUE_D": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "BLUE_E": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "BOLD": {"kind": "constant", "module": "", "signature": None},
        "BOOK": {"kind": "constant", "module": "", "signature": None},
        "BackgroundColoredVMobjectDisplayer": {
            "bases": ["object"],
            "kind": "class",
            "module": "manim.camera.camera",
            "signature": "(camera: 'Camera')",
        },
        "BackgroundRectangle": {
            "bases": [
                "SurroundingRectangle",
                "RoundedRectangle",
                "Rectangle",
                "Polygon",
                "Polygram",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.geometry.shape_matchers",
            "signature": "(*mobjects: 'Mobject', color: "
            "'ParsableManimColor | None' = None, "
            "stroke_width: 'float' = 0, stroke_opacity: "
            "'float' = 0, fill_opacity: 'float' = 0.75, "
            "buff: 'float | tuple[float, float]' = 0, "
            "**kwargs: 'Any') -> 'None'",
        },
        "BarChart": {
            "bases": ["Axes", "VGroup", "VMobject", "Mobject", "CoordinateSystem", "object"],
            "kind": "class",
            "module": "manim.mobject.graphing.probability",
            "signature": "(values: 'MutableSequence[float]', bar_names: "
            "'Sequence[str] | None' = None, y_range: 'Sequence[float] | "
            "None' = None, x_length: 'float | None' = None, y_length: "
            "'float | None' = None, bar_colors: 'Iterable[str]' = "
            "['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'], "
            "bar_width: 'float' = 0.6, bar_fill_opacity: 'float' = 0.7, "
            "bar_stroke_width: 'float' = 3, **kwargs: 'Any')",
        },
        "Blink": {
            "bases": ["Succession", "AnimationGroup", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.indication",
            "signature": "(mobject: 'Mobject', time_on: 'float' = 0.5, time_off: "
            "'float' = 0.5, blinks: 'int' = 1, hide_at_end: 'bool' = "
            "False, **kwargs: 'Any')",
        },
        "Brace": {
            "bases": ["VMobjectFromSVGPath", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.svg.brace",
            "signature": "(mobject: 'Mobject', direction: 'Vector3DLike' = array([ 0., "
            "-1.,  0.]), buff: 'float' = 0.2, sharpness: 'float' = 2, "
            "stroke_width: 'float' = 0, fill_opacity: 'float' = 1.0, "
            "background_stroke_width: 'float' = 0, "
            "background_stroke_color: 'ParsableManimColor' = "
            "ManimColor('#000000'), **kwargs: 'Any')",
        },
        "BraceBetweenPoints": {
            "bases": ["Brace", "VMobjectFromSVGPath", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.svg.brace",
            "signature": "(point_1: 'Point3DLike', point_2: 'Point3DLike', "
            "direction: 'Vector3DLike' = array([0., 0., 0.]), "
            "**kwargs: 'Any')",
        },
        "BraceLabel": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.svg.brace",
            "signature": "(obj: 'Mobject', text: 'str', brace_direction: "
            "'Vector3DLike' = array([ 0., -1.,  0.]), "
            "label_constructor: 'type[SingleStringMathTex | Text]' = "
            "<class 'manim.mobject.text.tex_mobject.MathTex'>, "
            "font_size: 'float' = 48, buff: 'float' = 0.2, "
            "brace_config: 'dict[str, Any] | None' = None, **kwargs: "
            "'Any')",
        },
        "BraceText": {
            "bases": ["BraceLabel", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.svg.brace",
            "signature": "(obj: 'Mobject', text: 'str', label_constructor: "
            "'type[SingleStringMathTex | Text]' = <class "
            "'manim.mobject.text.text_mobject.Text'>, **kwargs: "
            "'Any')",
        },
        "Broadcast": {
            "bases": ["LaggedStart", "AnimationGroup", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.specialized",
            "signature": "(mobject: 'Mobject', focal_point: 'Sequence[float]' = "
            "array([0., 0., 0.]), n_mobs: 'int' = 5, initial_opacity: "
            "'float' = 1, final_opacity: 'float' = 0, initial_width: "
            "'float' = 0.0, remover: 'bool' = True, lag_ratio: 'float' "
            "= 0.2, run_time: 'float' = 3, **kwargs: 'Any')",
        },
        "BulletedList": {
            "bases": [
                "Tex",
                "MathTex",
                "SingleStringMathTex",
                "SVGMobject",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.text.tex_mobject",
            "signature": "(*items: 'str', buff: 'float' = 0.5, dot_scale_factor: "
            "'float' = 2, tex_environment: 'str | None' = None, "
            "**kwargs: 'Any')",
        },
        "CHOOSE_NUMBER_MESSAGE": {"kind": "constant", "module": "", "signature": None},
        "CONTEXT_SETTINGS": {"kind": "constant", "module": "", "signature": None},
        "CTRL_VALUE": {"kind": "constant", "module": "", "signature": None},
        "CairoRenderer": {
            "bases": ["object"],
            "kind": "class",
            "module": "manim.renderer.cairo_renderer",
            "signature": "(file_writer_class: 'type[SceneFileWriter]' = <class "
            "'manim.scene.scene_file_writer.SceneFileWriter'>, "
            "camera_class: 'type[Camera] | None' = None, "
            "skip_animations: 'bool' = False, **kwargs: 'Any')",
        },
        "Camera": {
            "bases": ["object"],
            "kind": "class",
            "module": "manim.camera.camera",
            "signature": "(background_image: 'str | None' = None, frame_center: "
            "'Point3D' = array([0., 0., 0.]), image_mode: 'str' = 'RGBA', "
            "n_channels: 'int' = 4, pixel_array_dtype: 'str' = 'uint8', "
            "cairo_line_width_multiple: 'float' = 0.01, use_z_index: "
            "'bool' = True, background: 'PixelArray | None' = None, "
            "pixel_height: 'int | None' = None, pixel_width: 'int | None' "
            "= None, frame_height: 'float | None' = None, frame_width: "
            "'float | None' = None, frame_rate: 'float | None' = None, "
            "background_color: 'ParsableManimColor | None' = None, "
            "background_opacity: 'float | None' = None, **kwargs: 'Any') "
            "-> 'None'",
        },
        "CapStyleType": {
            "bases": ["Enum", "object"],
            "kind": "class",
            "module": "manim.constants",
            "signature": "(*values)",
        },
        "ChangeDecimalToValue": {
            "bases": ["ChangingDecimal", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.numbers",
            "signature": "(decimal_mob: 'DecimalNumber', target_number: "
            "'int', **kwargs: 'Any') -> 'None'",
        },
        "ChangeSpeed": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.speedmodifier",
            "signature": "(anim: 'Animation | _AnimationBuilder', speedinfo: "
            "'dict[float, float]', rate_func: 'Callable[[float], "
            "float] | None' = None, affects_speed_updaters: 'bool' = "
            "True, **kwargs) -> 'None'",
        },
        "ChangingDecimal": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.numbers",
            "signature": "(decimal_mob: 'DecimalNumber', number_update_func: "
            "'Callable[[float], float]', "
            "suspend_mobject_updating: 'bool' = False, **kwargs: "
            "'Any') -> 'None'",
        },
        "Circle": {
            "bases": ["Arc", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(radius: 'float | None' = None, color: 'ParsableManimColor' "
            "= ManimColor('#FC6255'), **kwargs: 'Any') -> 'None'",
        },
        "Circumscribe": {
            "bases": ["Succession", "AnimationGroup", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.indication",
            "signature": "(mobject: 'Mobject', shape: 'type[Rectangle] | "
            "type[Circle]' = <class "
            "'manim.mobject.geometry.polygram.Rectangle'>, fade_in: "
            "'bool' = False, fade_out: 'bool' = False, time_width: "
            "'float' = 0.3, buff: 'float' = 0.1, color: "
            "'ParsableManimColor' = ManimColor('#FFFF00'), "
            "run_time: 'float' = 1, stroke_width: 'float' = 4, "
            "**kwargs: 'Any')",
        },
        "ClockwiseTransform": {
            "bases": ["Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(mobject: 'Mobject', target_mobject: 'Mobject', "
            "path_arc: 'float' = -3.141592653589793, "
            "**kwargs) -> 'None'",
        },
        "Code": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.text.code_mobject",
            "signature": "(code_file: 'StrPath | None' = None, code_string: 'str | None' "
            "= None, language: 'str | None' = None, formatter_style: 'str' "
            "= 'vim', tab_width: 'int' = 4, add_line_numbers: 'bool' = "
            "True, line_numbers_from: 'int' = 1, background: "
            "\"Literal['rectangle', 'window']\" = 'rectangle', "
            "background_config: 'dict[str, Any] | None' = None, "
            "paragraph_config: 'dict[str, Any] | None' = None)",
        },
        "ComplexHomotopy": {
            "bases": ["Homotopy", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.movement",
            "signature": "(complex_homotopy: 'Callable[[complex, float], "
            "float]', mobject: 'Mobject', **kwargs: 'Any')",
        },
        "ComplexPlane": {
            "bases": [
                "NumberPlane",
                "Axes",
                "VGroup",
                "VMobject",
                "Mobject",
                "CoordinateSystem",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.graphing.coordinate_systems",
            "signature": "(**kwargs: 'Any')",
        },
        "ComplexValueTracker": {
            "bases": ["ValueTracker", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.value_tracker",
            "signature": "(value: 'float' = 0, **kwargs: 'Any') -> 'None'",
        },
        "Cone": {
            "bases": ["Surface", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.three_dimensions",
            "signature": "(base_radius: 'float' = 1, height: 'float' = 1, direction: "
            "'Vector3DLike' = array([0., 0., 1.]), show_base: 'bool' = "
            "False, v_range: 'tuple[float, float]' = (0, "
            "6.283185307179586), u_min: 'float' = 0, checkerboard_colors: "
            "'Iterable[ParsableManimColor] | Literal[False]' = False, "
            "**kwargs: 'Any') -> 'None'",
        },
        "ConvexHull": {
            "bases": ["Polygram", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.polygram",
            "signature": "(*points: 'Point3DLike', tolerance: 'float' = 1e-05, "
            "**kwargs: 'Any') -> 'None'",
        },
        "ConvexHull3D": {
            "bases": ["Polyhedron", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.polyhedra",
            "signature": "(*points: 'Point3D', tolerance: 'float' = 1e-05, **kwargs: 'Any')",
        },
        "CoordinateSystem": {
            "bases": ["object"],
            "kind": "class",
            "module": "manim.mobject.graphing.coordinate_systems",
            "signature": "(x_range: 'Sequence[float] | None' = None, "
            "y_range: 'Sequence[float] | None' = None, "
            "x_length: 'float | None' = None, y_length: 'float "
            "| None' = None, dimension: 'int' = 2)",
        },
        "CounterclockwiseTransform": {
            "bases": ["Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(mobject: 'Mobject', target_mobject: "
            "'Mobject', path_arc: 'float' = "
            "3.141592653589793, **kwargs) -> 'None'",
        },
        "Create": {
            "bases": ["ShowPartial", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.creation",
            "signature": "(mobject: 'VMobject | OpenGLVMobject | OpenGLSurface', "
            "lag_ratio: 'float' = 1.0, introducer: 'bool' = True, "
            "**kwargs) -> 'None'",
        },
        "Cross": {
            "bases": ["VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.shape_matchers",
            "signature": "(mobject: 'Mobject | None' = None, stroke_color: "
            "'ParsableManimColor' = ManimColor('#FC6255'), stroke_width: "
            "'float' = 6.0, scale_factor: 'float' = 1.0, **kwargs: 'Any') "
            "-> 'None'",
        },
        "Cube": {
            "bases": ["VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.three_dimensions",
            "signature": "(side_length: 'float' = 2, fill_opacity: 'float' = 0.75, "
            "fill_color: 'ParsableManimColor' = ManimColor('#58C4DD'), "
            "stroke_width: 'float' = 0, **kwargs: 'Any') -> 'None'",
        },
        "CubicBezier": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(start_anchor: 'Point3DLike', start_handle: "
            "'Point3DLike', end_handle: 'Point3DLike', end_anchor: "
            "'Point3DLike', **kwargs: 'Any') -> 'None'",
        },
        "CurvedArrow": {
            "bases": [
                "ArcBetweenPoints",
                "Arc",
                "TipableVMobject",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(start_point: 'Point3DLike', end_point: 'Point3DLike', "
            "**kwargs: 'Any') -> 'None'",
        },
        "CurvedDoubleArrow": {
            "bases": [
                "CurvedArrow",
                "ArcBetweenPoints",
                "Arc",
                "TipableVMobject",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(start_point: 'Point3DLike', end_point: "
            "'Point3DLike', **kwargs: 'Any') -> 'None'",
        },
        "CurvesAsSubmobjects": {
            "bases": ["VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.types.vectorized_mobject",
            "signature": "(vmobject: 'VMobject', **kwargs) -> 'None'",
        },
        "Cutout": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.polygram",
            "signature": "(main_shape: 'VMobject', *mobjects: 'VMobject', **kwargs: "
            "'Any') -> 'None'",
        },
        "CyclicReplace": {
            "bases": ["Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(*mobjects: 'Mobject', path_arc: 'float' = "
            "1.5707963267948966, **kwargs) -> 'None'",
        },
        "Cylinder": {
            "bases": ["Surface", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.three_dimensions",
            "signature": "(radius: 'float' = 1, height: 'float' = 2, direction: "
            "'Vector3DLike' = array([0., 0., 1.]), v_range: "
            "'tuple[float, float]' = (0, 6.283185307179586), show_ends: "
            "'bool' = True, resolution: 'int | tuple[int, int]' = (24, "
            "24), **kwargs: 'Any') -> 'None'",
        },
        "DARKER_GRAY": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "DARKER_GREY": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "DARK_BLUE": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "DARK_BROWN": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "DARK_GRAY": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "DARK_GREY": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "DEFAULT_ARROW_TIP_LENGTH": {"kind": "constant", "module": "", "signature": None},
        "DEFAULT_DASH_LENGTH": {"kind": "constant", "module": "", "signature": None},
        "DEFAULT_DOT_RADIUS": {"kind": "constant", "module": "", "signature": None},
        "DEFAULT_FONT_SIZE": {"kind": "constant", "module": "", "signature": None},
        "DEFAULT_MOBJECT_TO_EDGE_BUFFER": {"kind": "constant", "module": "", "signature": None},
        "DEFAULT_MOBJECT_TO_MOBJECT_BUFFER": {"kind": "constant", "module": "", "signature": None},
        "DEFAULT_POINTWISE_FUNCTION_RUN_TIME": {
            "kind": "constant",
            "module": "",
            "signature": None,
        },
        "DEFAULT_POINT_DENSITY_1D": {"kind": "constant", "module": "", "signature": None},
        "DEFAULT_POINT_DENSITY_2D": {"kind": "constant", "module": "", "signature": None},
        "DEFAULT_QUALITY": {"kind": "constant", "module": "", "signature": None},
        "DEFAULT_SMALL_DOT_RADIUS": {"kind": "constant", "module": "", "signature": None},
        "DEFAULT_STROKE_WIDTH": {"kind": "constant", "module": "", "signature": None},
        "DEFAULT_WAIT_TIME": {"kind": "constant", "module": "", "signature": None},
        "DEGREES": {"kind": "constant", "module": "", "signature": None},
        "DL": {"kind": "constant", "module": "", "signature": None},
        "DOWN": {"kind": "constant", "module": "", "signature": None},
        "DR": {"kind": "constant", "module": "", "signature": None},
        "DashedLine": {
            "bases": ["Line", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.line",
            "signature": "(*args: 'Any', dash_length: 'float' = 0.05, "
            "dashed_ratio: 'float' = 0.5, **kwargs: 'Any') -> 'None'",
        },
        "DashedVMobject": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.types.vectorized_mobject",
            "signature": "(vmobject: 'VMobject', num_dashes: 'int' = 15, "
            "dashed_ratio: 'float' = 0.5, dash_offset: 'float' = "
            "0, color: 'ManimColor' = ManimColor('#FFFFFF'), "
            "equal_lengths: 'bool' = True, **kwargs) -> 'None'",
        },
        "DecimalMatrix": {
            "bases": ["Matrix", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.matrix",
            "signature": "(matrix: 'Iterable', element_to_mobject: "
            "'type[Mobject]' = <class "
            "'manim.mobject.text.numbers.DecimalNumber'>, "
            "element_to_mobject_config: 'dict[str, Any]' = "
            "{'num_decimal_places': 1}, **kwargs: 'Any')",
        },
        "DecimalNumber": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.text.numbers",
            "signature": "(number: 'float' = 0, num_decimal_places: 'int' = 2, "
            "mob_class: 'type[SingleStringMathTex]' = <class "
            "'manim.mobject.text.tex_mobject.MathTex'>, "
            "include_sign: 'bool' = False, group_with_commas: "
            "'bool' = True, digit_buff_per_font_unit: 'float' = "
            "0.001, show_ellipsis: 'bool' = False, unit: 'str | "
            "None' = None, unit_buff_per_font_unit: 'float' = 0, "
            "include_background_rectangle: 'bool' = False, "
            "edge_to_fix: 'Vector3DLike' = array([-1.,  0.,  0.]), "
            "font_size: 'float' = 48, stroke_width: 'float' = 0, "
            "fill_opacity: 'float' = 1.0, **kwargs: 'Any')",
        },
        "DecimalTable": {
            "bases": ["Table", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.table",
            "signature": "(table: 'Iterable[Iterable[float | str]]', "
            "element_to_mobject: 'Callable[[float | str], "
            "VMobject]' = <class "
            "'manim.mobject.text.numbers.DecimalNumber'>, "
            "element_to_mobject_config: 'dict' = "
            "{'num_decimal_places': 1}, **kwargs)",
        },
        "DefaultSectionType": {
            "bases": ["str", "Enum", "object"],
            "kind": "class",
            "module": "manim.scene.section",
            "signature": "(*values)",
        },
        "DiGraph": {
            "bases": ["GenericGraph", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.graph",
            "signature": "(vertices: 'Sequence[Hashable]', edges: "
            "'Sequence[tuple[Hashable, Hashable]]', labels: 'bool | "
            "dict' = False, label_fill_color: 'str' = "
            "ManimColor('#000000'), layout: 'LayoutName | dict[Hashable, "
            "Point3DLike] | LayoutFunction' = 'spring', layout_scale: "
            "'float | tuple[float, float, float]' = 2, layout_config: "
            "'dict | None' = None, vertex_type: 'type[Mobject]' = <class "
            "'manim.mobject.geometry.arc.Dot'>, vertex_config: 'dict | "
            "None' = None, vertex_mobjects: 'dict | None' = None, "
            "edge_type: 'type[Mobject]' = <class "
            "'manim.mobject.geometry.line.Line'>, partitions: "
            "'Sequence[Sequence[Hashable]] | None' = None, root_vertex: "
            "'Hashable | None' = None, edge_config: 'dict | None' = "
            "None) -> 'None'",
        },
        "DictAsObject": {
            "bases": ["object"],
            "kind": "class",
            "module": "manim.utils.config_ops",
            "signature": "(dictin: 'dict[str, Any]')",
        },
        "Difference": {
            "bases": ["_BooleanOps", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.boolean_ops",
            "signature": "(subject: 'VMobject', clip: 'VMobject', **kwargs: 'Any') -> 'None'",
        },
        "Dodecahedron": {
            "bases": ["Polyhedron", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.polyhedra",
            "signature": "(edge_length: 'float' = 1, **kwargs: 'Any')",
        },
        "Dot": {
            "bases": ["Circle", "Arc", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(point: 'Point3DLike' = array([0., 0., 0.]), radius: 'float' = "
            "0.08, stroke_width: 'float' = 0, fill_opacity: 'float' = 1.0, "
            "color: 'ParsableManimColor' = ManimColor('#FFFFFF'), **kwargs: "
            "'Any') -> 'None'",
        },
        "Dot3D": {
            "bases": ["Sphere", "Surface", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.three_dimensions",
            "signature": "(point: 'Point3D' = array([0., 0., 0.]), radius: 'float' = "
            "0.08, color: 'ParsableManimColor' = ManimColor('#FFFFFF'), "
            "resolution: 'int | tuple[int, int] | None' = (8, 8), "
            "**kwargs: 'Any') -> 'None'",
        },
        "DotCloud": {
            "bases": ["OpenGLPMobject", "OpenGLMobject", "object"],
            "kind": "class",
            "module": "manim.mobject.opengl.dot_cloud",
            "signature": "(color: 'ParsableManimColor' = ManimColor('#FFFF00'), "
            "stroke_width: 'float' = 2.0, radius: 'float' = 2.0, "
            "density: 'float' = 10, **kwargs: 'Any')",
        },
        "DoubleArrow": {
            "bases": ["Arrow", "Line", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.line",
            "signature": "(*args: 'Any', **kwargs: 'Any') -> 'None'",
        },
        "DrawBorderThenFill": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.creation",
            "signature": "(vmobject: 'VMobject | OpenGLVMobject', "
            "run_time: 'float' = 2, rate_func: "
            "'Callable[[float], float]' = <function "
            "double_smooth>, stroke_width: 'float' = 2, "
            "stroke_color: 'str' = None, introducer: 'bool' = "
            "True, **kwargs) -> 'None'",
        },
        "EPILOG": {"kind": "constant", "module": "", "signature": None},
        "Elbow": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.line",
            "signature": "(width: 'float' = 0.2, angle: 'float' = 0, **kwargs: 'Any') -> 'None'",
        },
        "Ellipse": {
            "bases": ["Circle", "Arc", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(width: 'float' = 2, height: 'float' = 1, **kwargs: 'Any') -> 'None'",
        },
        "Exclusion": {
            "bases": ["_BooleanOps", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.boolean_ops",
            "signature": "(subject: 'VMobject', clip: 'VMobject', **kwargs: 'Any') -> 'None'",
        },
        "FadeIn": {
            "bases": ["_Fade", "Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.fading",
            "signature": "(*mobjects: 'Mobject', **kwargs: 'Any') -> 'None'",
        },
        "FadeOut": {
            "bases": ["_Fade", "Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.fading",
            "signature": "(*mobjects: 'Mobject', **kwargs: 'Any') -> 'None'",
        },
        "FadeToColor": {
            "bases": ["ApplyMethod", "Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(mobject: 'Mobject', color: 'str', **kwargs) -> 'None'",
        },
        "FadeTransform": {
            "bases": ["Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(mobject, target_mobject, stretch=True, dim_to_match=1, **kwargs)",
        },
        "FadeTransformPieces": {
            "bases": ["FadeTransform", "Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(mobject, target_mobject, stretch=True, dim_to_match=1, **kwargs)",
        },
        "Flash": {
            "bases": ["AnimationGroup", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.indication",
            "signature": "(point: 'Point3DLike | Mobject', line_length: 'float' = 0.2, "
            "num_lines: 'int' = 12, flash_radius: 'float' = 0.1, "
            "line_stroke_width: 'int' = 3, color: 'ParsableManimColor' = "
            "ManimColor('#FFFF00'), time_width: 'float' = 1, run_time: "
            "'float' = 1.0, **kwargs: 'Any')",
        },
        "FocusOn": {
            "bases": ["Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.indication",
            "signature": "(focus_point: 'Point3DLike | Mobject', opacity: 'float' = "
            "0.2, color: 'ParsableManimColor' = ManimColor('#888888'), "
            "run_time: 'float' = 2, **kwargs: 'Any')",
        },
        "FullScreenRectangle": {
            "bases": [
                "ScreenRectangle",
                "Rectangle",
                "Polygon",
                "Polygram",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.frame",
            "signature": "(**kwargs: 'Any') -> 'None'",
        },
        "FunctionGraph": {
            "bases": ["ParametricFunction", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.graphing.functions",
            "signature": "(function: 'Callable[[float], Any]', x_range: "
            "'tuple[float, float] | tuple[float, float, float] | "
            "None' = None, color: 'ParsableManimColor' = "
            "ManimColor('#FFFF00'), **kwargs: 'Any') -> 'None'",
        },
        "GOLD": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GOLD_A": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GOLD_B": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GOLD_C": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GOLD_D": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GOLD_E": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GRAY": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GRAY_A": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GRAY_B": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GRAY_BROWN": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GRAY_C": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GRAY_D": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GRAY_E": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GREEN": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GREEN_A": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GREEN_B": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GREEN_C": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GREEN_D": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GREEN_E": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GREY": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GREY_A": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GREY_B": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GREY_BROWN": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GREY_C": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GREY_D": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "GREY_E": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "Graph": {
            "bases": ["GenericGraph", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.graph",
            "signature": "(vertices: 'Sequence[Hashable]', edges: "
            "'Sequence[tuple[Hashable, Hashable]]', labels: 'bool | dict' "
            "= False, label_fill_color: 'str' = ManimColor('#000000'), "
            "layout: 'LayoutName | dict[Hashable, Point3DLike] | "
            "LayoutFunction' = 'spring', layout_scale: 'float | "
            "tuple[float, float, float]' = 2, layout_config: 'dict | None' "
            "= None, vertex_type: 'type[Mobject]' = <class "
            "'manim.mobject.geometry.arc.Dot'>, vertex_config: 'dict | "
            "None' = None, vertex_mobjects: 'dict | None' = None, "
            "edge_type: 'type[Mobject]' = <class "
            "'manim.mobject.geometry.line.Line'>, partitions: "
            "'Sequence[Sequence[Hashable]] | None' = None, root_vertex: "
            "'Hashable | None' = None, edge_config: 'dict | None' = None) "
            "-> 'None'",
        },
        "Group": {
            "bases": ["Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.mobject",
            "signature": "(*mobjects, **kwargs) -> 'None'",
        },
        "GrowArrow": {
            "bases": ["GrowFromPoint", "Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.growing",
            "signature": "(arrow: 'Arrow', point_color: 'ParsableManimColor | None' "
            "= None, **kwargs: 'Any')",
        },
        "GrowFromCenter": {
            "bases": ["GrowFromPoint", "Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.growing",
            "signature": "(mobject: 'Mobject', point_color: "
            "'ParsableManimColor | None' = None, **kwargs: "
            "'Any')",
        },
        "GrowFromEdge": {
            "bases": ["GrowFromPoint", "Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.growing",
            "signature": "(mobject: 'Mobject', edge: 'Vector3DLike', "
            "point_color: 'ParsableManimColor | None' = None, "
            "**kwargs: 'Any')",
        },
        "GrowFromPoint": {
            "bases": ["Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.growing",
            "signature": "(mobject: 'Mobject', point: 'Point3DLike', "
            "point_color: 'ParsableManimColor | None' = None, "
            "**kwargs: 'Any')",
        },
        "HEAVY": {"kind": "constant", "module": "", "signature": None},
        "HSV": {
            "bases": ["ManimColor", "object"],
            "kind": "class",
            "module": "manim.utils.color.core",
            "signature": "(hsv: 'FloatHSVLike | FloatHSVALike', alpha: 'float' = 1.0) -> 'None'",
        },
        "Homotopy": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.movement",
            "signature": "(homotopy: 'Callable[[float, float, float, float], "
            "tuple[float, float, float]]', mobject: 'Mobject', "
            "run_time: 'float' = 3, apply_function_kwargs: 'dict[str, "
            "Any] | None' = None, **kwargs: 'Any')",
        },
        "IN": {"kind": "constant", "module": "", "signature": None},
        "INVALID_NUMBER_MESSAGE": {"kind": "constant", "module": "", "signature": None},
        "ITALIC": {"kind": "constant", "module": "", "signature": None},
        "Icosahedron": {
            "bases": ["Polyhedron", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.polyhedra",
            "signature": "(edge_length: 'float' = 1, **kwargs: 'Any')",
        },
        "ImageMobject": {
            "bases": ["AbstractImageMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.types.image_mobject",
            "signature": "(filename_or_array: 'StrPath | npt.NDArray', "
            "scale_to_resolution: 'int' = 1080, invert: 'bool' = "
            "False, image_mode: 'str' = 'RGBA', **kwargs: 'Any') -> "
            "'None'",
        },
        "ImageMobjectFromCamera": {
            "bases": ["AbstractImageMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.types.image_mobject",
            "signature": "(camera: 'MovingCamera', "
            "default_display_frame_config: 'dict[str, "
            "Any] | None' = None, **kwargs: 'Any') -> "
            "'None'",
        },
        "ImplicitFunction": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.graphing.functions",
            "signature": "(func: 'Callable[[float, float], float]', x_range: "
            "'Sequence[float] | None' = None, y_range: "
            "'Sequence[float] | None' = None, min_depth: 'int' "
            "= 5, max_quads: 'int' = 1500, use_smoothing: "
            "'bool' = True, **kwargs: 'Any')",
        },
        "Indicate": {
            "bases": ["Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.indication",
            "signature": "(mobject: 'Mobject', scale_factor: 'float' = 1.2, color: "
            "'ParsableManimColor' = ManimColor('#FFFF00'), rate_func: "
            "'RateFunction' = <function there_and_back>, **kwargs: "
            "'Any')",
        },
        "Integer": {
            "bases": ["DecimalNumber", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.text.numbers",
            "signature": "(number: 'float' = 0, num_decimal_places: 'int' = 0, "
            "**kwargs: 'Any') -> 'None'",
        },
        "IntegerMatrix": {
            "bases": ["Matrix", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.matrix",
            "signature": "(matrix: 'Iterable', element_to_mobject: "
            "'type[Mobject]' = <class "
            "'manim.mobject.text.numbers.Integer'>, **kwargs: "
            "'Any')",
        },
        "IntegerTable": {
            "bases": ["Table", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.table",
            "signature": "(table: 'Iterable[Iterable[float | str]]', "
            "element_to_mobject: 'Callable[[float | str], "
            "VMobject]' = <class "
            "'manim.mobject.text.numbers.Integer'>, **kwargs)",
        },
        "Intersection": {
            "bases": ["_BooleanOps", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.boolean_ops",
            "signature": "(*vmobjects: 'VMobject', **kwargs: 'Any') -> 'None'",
        },
        "LARGE_BUFF": {"kind": "constant", "module": "", "signature": None},
        "LEFT": {"kind": "constant", "module": "", "signature": None},
        "LIGHT": {"kind": "constant", "module": "", "signature": None},
        "LIGHTER_GRAY": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "LIGHTER_GREY": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "LIGHT_BROWN": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "LIGHT_GRAY": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "LIGHT_GREY": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "LIGHT_PINK": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "LOGO_BLACK": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "LOGO_BLUE": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "LOGO_GREEN": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "LOGO_RED": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "LOGO_WHITE": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "Label": {
            "bases": ["VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.labeled",
            "signature": "(label: 'str | Tex | MathTex | Text', label_config: "
            "'dict[str, Any] | None' = None, box_config: 'dict[str, Any] | "
            "None' = None, frame_config: 'dict[str, Any] | None' = None, "
            "**kwargs: 'Any') -> 'None'",
        },
        "LabeledArrow": {
            "bases": [
                "LabeledLine",
                "Arrow",
                "Line",
                "TipableVMobject",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.geometry.labeled",
            "signature": "(*args: 'Any', **kwargs: 'Any') -> 'None'",
        },
        "LabeledDot": {
            "bases": ["Dot", "Circle", "Arc", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(label: 'str | SingleStringMathTex | Text | Tex', "
            "radius: 'float | None' = None, buff: 'float' = 0.1, "
            "**kwargs: 'Any') -> 'None'",
        },
        "LabeledLine": {
            "bases": ["Line", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.labeled",
            "signature": "(label: 'str | Tex | MathTex | Text', label_position: "
            "'float' = 0.5, label_config: 'dict[str, Any] | None' = "
            "None, box_config: 'dict[str, Any] | None' = None, "
            "frame_config: 'dict[str, Any] | None' = None, *args: "
            "'Any', **kwargs: 'Any') -> 'None'",
        },
        "LabeledPolygram": {
            "bases": ["Polygram", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.labeled",
            "signature": "(*vertex_groups: 'Point3DLike_Array', label: 'str | "
            "Tex | MathTex | Text', precision: 'float' = 0.01, "
            "label_config: 'dict[str, Any] | None' = None, "
            "box_config: 'dict[str, Any] | None' = None, "
            "frame_config: 'dict[str, Any] | None' = None, "
            "**kwargs: 'Any') -> 'None'",
        },
        "LaggedStart": {
            "bases": ["AnimationGroup", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.composition",
            "signature": "(*animations: 'Animation', lag_ratio: 'float' = 0.05, **kwargs: 'Any')",
        },
        "LaggedStartMap": {
            "bases": ["LaggedStart", "AnimationGroup", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.composition",
            "signature": "(animation_class: 'type[Animation]', mobject: "
            "'Mobject', arg_creator: 'Callable[[Mobject], "
            "Iterable[Any]] | None' = None, run_time: 'float' = "
            "2, lag_ratio: 'float' = 0.05, **kwargs: 'Any')",
        },
        "Line": {
            "bases": ["TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.line",
            "signature": "(start: 'Point3DLike | Mobject' = array([-1.,  0.,  0.]), end: "
            "'Point3DLike | Mobject' = array([1., 0., 0.]), buff: 'float' = "
            "0, path_arc: 'float' = 0, **kwargs: 'Any') -> 'None'",
        },
        "Line3D": {
            "bases": ["Cylinder", "Surface", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.three_dimensions",
            "signature": "(start: 'Point3DLike' = array([-1.,  0.,  0.]), end: "
            "'Point3DLike' = array([1., 0., 0.]), thickness: 'float' = "
            "0.02, color: 'ParsableManimColor | None' = None, resolution: "
            "'int | tuple[int, int]' = 24, **kwargs: 'Any')",
        },
        "LineJointType": {
            "bases": ["Enum", "object"],
            "kind": "class",
            "module": "manim.constants",
            "signature": "(*values)",
        },
        "LinearBase": {
            "bases": ["_ScaleBase", "object"],
            "kind": "class",
            "module": "manim.mobject.graphing.scale",
            "signature": "(scale_factor: 'float' = 1.0)",
        },
        "LinearTransformationScene": {
            "bases": ["VectorScene", "Scene", "object"],
            "kind": "class",
            "module": "manim.scene.vector_space_scene",
            "signature": "(include_background_plane: 'bool' = True, "
            "include_foreground_plane: 'bool' = True, "
            "background_plane_kwargs: 'dict[str, Any] "
            "| None' = None, foreground_plane_kwargs: "
            "'dict[str, Any] | None' = None, "
            "show_coordinates: 'bool' = False, "
            "show_basis_vectors: 'bool' = True, "
            "basis_vector_stroke_width: 'float' = 6, "
            "i_hat_color: 'ParsableManimColor' = "
            "ManimColor('#83C167'), j_hat_color: "
            "'ParsableManimColor' = "
            "ManimColor('#FC6255'), "
            "leave_ghost_vectors: 'bool' = False, "
            "**kwargs: 'Any') -> 'None'",
        },
        "LogBase": {
            "bases": ["_ScaleBase", "object"],
            "kind": "class",
            "module": "manim.mobject.graphing.scale",
            "signature": "(base: 'float' = 10, custom_labels: 'bool' = True)",
        },
        "MAROON": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "MAROON_A": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "MAROON_B": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "MAROON_C": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "MAROON_D": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "MAROON_E": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "MEDIUM": {"kind": "constant", "module": "", "signature": None},
        "MED_LARGE_BUFF": {"kind": "constant", "module": "", "signature": None},
        "MED_SMALL_BUFF": {"kind": "constant", "module": "", "signature": None},
        "MaintainPositionRelativeTo": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.updaters.update",
            "signature": "(mobject: 'Mobject', tracked_mobject: "
            "'Mobject', **kwargs: 'Any') -> 'None'",
        },
        "ManimBanner": {
            "bases": ["VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.logo",
            "signature": "(dark_theme: 'bool' = True)",
        },
        "ManimColor": {
            "bases": ["object"],
            "kind": "class",
            "module": "manim.utils.color.core",
            "signature": "(value: 'ParsableManimColor | None', alpha: 'float' = 1.0) -> 'None'",
        },
        "MappingCamera": {
            "bases": ["Camera", "object"],
            "kind": "class",
            "module": "manim.camera.mapping_camera",
            "signature": "(mapping_func=<function MappingCamera.<lambda>>, "
            "min_num_curves=50, allow_object_intrusion=False, "
            "**kwargs)",
        },
        "MarkupText": {
            "bases": ["SVGMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.text.text_mobject",
            "signature": "(text: 'str', fill_opacity: 'float' = 1, stroke_width: "
            "'float' = 0, color: 'ParsableManimColor | None' = None, "
            "font_size: 'float' = 48, line_spacing: 'float' = -1, "
            "font: 'str' = '', slant: 'str' = 'NORMAL', weight: 'str' "
            "= 'NORMAL', justify: 'bool' = False, gradient: "
            "'Iterable[ParsableManimColor] | None' = None, tab_width: "
            "'int' = 4, height: 'int | None' = None, width: 'int | "
            "None' = None, should_center: 'bool' = True, "
            "disable_ligatures: 'bool' = False, warn_missing_font: "
            "'bool' = True, **kwargs: 'Any')",
        },
        "MathTable": {
            "bases": ["Table", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.table",
            "signature": "(table: 'Iterable[Iterable[float | str]]', "
            "element_to_mobject: 'Callable[[float | str], VMobject]' = "
            "<class 'manim.mobject.text.tex_mobject.MathTex'>, "
            "**kwargs)",
        },
        "MathTex": {
            "bases": ["SingleStringMathTex", "SVGMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.text.tex_mobject",
            "signature": "(*tex_strings: 'str', arg_separator: 'str' = ' ', "
            "substrings_to_isolate: 'Iterable[str] | None' = None, "
            "tex_to_color_map: 'dict[str, ParsableManimColor] | None' = "
            "None, tex_environment: 'str | None' = 'align*', **kwargs: "
            "'Any')",
        },
        "Matrix": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.matrix",
            "signature": "(matrix: 'Iterable', v_buff: 'float' = 0.8, h_buff: 'float' "
            "= 1.3, bracket_h_buff: 'float' = 0.25, bracket_v_buff: "
            "'float' = 0.25, add_background_rectangles_to_entries: 'bool' "
            "= False, include_background_rectangle: 'bool' = False, "
            "element_to_mobject: 'type[Mobject] | Callable[..., Mobject]' "
            "= <class 'manim.mobject.text.tex_mobject.MathTex'>, "
            "element_to_mobject_config: 'dict' = {}, "
            "element_alignment_corner: 'Sequence[float]' = array([ 1., "
            "-1.,  0.]), left_bracket: 'str' = '[', right_bracket: 'str' "
            "= ']', stretch_brackets: 'bool' = True, bracket_config: "
            "'dict' = {}, **kwargs: 'Any')",
        },
        "Mobject": {
            "bases": ["object"],
            "kind": "class",
            "module": "manim.mobject.mobject",
            "signature": "(color: 'ParsableManimColor | list[ParsableManimColor]' = "
            "ManimColor('#FFFFFF'), name: 'str | None' = None, dim: "
            "'int' = 3, target=None, z_index: 'float' = 0) -> 'None'",
        },
        "Mobject1D": {
            "bases": ["PMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.types.point_cloud_mobject",
            "signature": "(density: 'int' = 10, **kwargs: 'Any') -> 'None'",
        },
        "Mobject2D": {
            "bases": ["PMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.types.point_cloud_mobject",
            "signature": "(density: 'int' = 25, **kwargs: 'Any') -> 'None'",
        },
        "MobjectMatrix": {
            "bases": ["Matrix", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.matrix",
            "signature": "(matrix: 'Iterable', element_to_mobject: "
            "'type[Mobject] | Callable[..., Mobject]' = <function "
            "MobjectMatrix.<lambda>>, **kwargs: 'Any')",
        },
        "MobjectTable": {
            "bases": ["Table", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.table",
            "signature": "(table: 'Iterable[Iterable[VMobject]]', "
            "element_to_mobject: 'Callable[[VMobject], VMobject]' = "
            "<function MobjectTable.<lambda>>, **kwargs)",
        },
        "MoveAlongPath": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.movement",
            "signature": "(mobject: 'Mobject', path: 'VMobject', "
            "suspend_mobject_updating: 'bool' = False, **kwargs: "
            "'Any')",
        },
        "MoveToTarget": {
            "bases": ["Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(mobject: 'Mobject', **kwargs) -> 'None'",
        },
        "MovingCamera": {
            "bases": ["Camera", "object"],
            "kind": "class",
            "module": "manim.camera.moving_camera",
            "signature": "(frame: 'Mobject | None' = None, fixed_dimension: "
            "'int' = 0, default_frame_stroke_color: 'ManimColor' = "
            "ManimColor('#FFFFFF'), default_frame_stroke_width: "
            "'int' = 0, **kwargs: 'Any')",
        },
        "MovingCameraScene": {
            "bases": ["Scene", "object"],
            "kind": "class",
            "module": "manim.scene.moving_camera_scene",
            "signature": "(camera_class: 'type[Camera]' = <class "
            "'manim.camera.moving_camera.MovingCamera'>, "
            "**kwargs: 'Any') -> 'None'",
        },
        "MultiCamera": {
            "bases": ["MovingCamera", "Camera", "object"],
            "kind": "class",
            "module": "manim.camera.multi_camera",
            "signature": "(image_mobjects_from_cameras: "
            "'Iterable[ImageMobjectFromCamera] | None' = None, "
            "allow_cameras_to_capture_their_own_display: 'bool' = "
            "False, **kwargs: 'Any') -> 'None'",
        },
        "NORMAL": {"kind": "constant", "module": "", "signature": None},
        "NO_SCENE_MESSAGE": {"kind": "constant", "module": "", "signature": None},
        "NumberLine": {
            "bases": ["Line", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.graphing.number_line",
            "signature": "(x_range: 'Sequence[float] | None' = None, length: "
            "'float | None' = None, unit_size: 'float' = 1, "
            "include_ticks: 'bool' = True, tick_size: 'float' = 0.1, "
            "numbers_with_elongated_ticks: 'Iterable[float] | None' = "
            "None, longer_tick_multiple: 'int' = 2, "
            "exclude_origin_tick: 'bool' = False, rotation: 'float' = "
            "0, stroke_width: 'float' = 2.0, include_tip: 'bool' = "
            "False, tip_width: 'float' = 0.35, tip_height: 'float' = "
            "0.35, tip_shape: 'type[ArrowTip] | None' = None, "
            "include_numbers: 'bool' = False, font_size: 'float' = "
            "36, label_direction: 'Point3DLike' = array([ 0., -1.,  "
            "0.]), label_constructor: 'type[MathTex]' = <class "
            "'manim.mobject.text.tex_mobject.MathTex'>, scaling: "
            "'_ScaleBase' = <manim.mobject.graphing.scale.LinearBase "
            "object>, line_to_number_buff: 'float' = 0.25, "
            "decimal_number_config: 'dict | None' = None, "
            "numbers_to_exclude: 'Iterable[float] | None' = None, "
            "numbers_to_include: 'Iterable[float] | None' = None, "
            "**kwargs: 'Any')",
        },
        "NumberPlane": {
            "bases": ["Axes", "VGroup", "VMobject", "Mobject", "CoordinateSystem", "object"],
            "kind": "class",
            "module": "manim.mobject.graphing.coordinate_systems",
            "signature": "(x_range: 'Sequence[float] | None' = "
            "(-7.111111111111111, 7.111111111111111, 1), y_range: "
            "'Sequence[float] | None' = (-4.0, 4.0, 1), x_length: "
            "'float | None' = None, y_length: 'float | None' = None, "
            "background_line_style: 'dict[str, Any] | None' = None, "
            "faded_line_style: 'dict[str, Any] | None' = None, "
            "faded_line_ratio: 'int' = 1, "
            "make_smooth_after_applying_functions: 'bool' = True, "
            "**kwargs: 'dict[str, Any]')",
        },
        "OBLIQUE": {"kind": "constant", "module": "", "signature": None},
        "ORANGE": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "ORIGIN": {"kind": "constant", "module": "", "signature": None},
        "OUT": {"kind": "constant", "module": "", "signature": None},
        "Octahedron": {
            "bases": ["Polyhedron", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.polyhedra",
            "signature": "(edge_length: 'float' = 1, **kwargs: 'Any')",
        },
        "OldMultiCamera": {
            "bases": ["Camera", "object"],
            "kind": "class",
            "module": "manim.camera.mapping_camera",
            "signature": "(*cameras_with_start_positions, **kwargs)",
        },
        "OpenGLPGroup": {
            "bases": ["OpenGLPMobject", "OpenGLMobject", "object"],
            "kind": "class",
            "module": "manim.mobject.opengl.opengl_point_cloud_mobject",
            "signature": "(*pmobs, **kwargs)",
        },
        "OpenGLPMPoint": {
            "bases": ["OpenGLPMobject", "OpenGLMobject", "object"],
            "kind": "class",
            "module": "manim.mobject.opengl.opengl_point_cloud_mobject",
            "signature": "(location=array([0., 0., 0.]), stroke_width=4.0, **kwargs)",
        },
        "OpenGLPMobject": {
            "bases": ["OpenGLMobject", "object"],
            "kind": "class",
            "module": "manim.mobject.opengl.opengl_point_cloud_mobject",
            "signature": "(stroke_width: 'float' = 2.0, color: "
            "'ParsableManimColor' = ManimColor('#FFFF00'), "
            "render_primitive: 'int' = 0, **kwargs)",
        },
        "PGroup": {
            "bases": ["PMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.types.point_cloud_mobject",
            "signature": "(*pmobs: 'Any', **kwargs: 'Any') -> 'None'",
        },
        "PI": {"kind": "constant", "module": "", "signature": None},
        "PINK": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "PMobject": {
            "bases": ["Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.types.point_cloud_mobject",
            "signature": "(stroke_width: 'int' = 4, **kwargs: 'Any') -> 'None'",
        },
        "PURE_BLUE": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "PURE_CYAN": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "PURE_GREEN": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "PURE_MAGENTA": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "PURE_RED": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "PURE_YELLOW": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "PURPLE": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "PURPLE_A": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "PURPLE_B": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "PURPLE_C": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "PURPLE_D": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "PURPLE_E": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "Paragraph": {
            "bases": ["VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.text.text_mobject",
            "signature": "(*text: 'str', line_spacing: 'float' = -1, alignment: "
            "'str | None' = None, **kwargs: 'Any')",
        },
        "ParametricFunction": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.graphing.functions",
            "signature": "(function: 'Callable[[float], Point3DLike]', "
            "t_range: 'tuple[float, float] | tuple[float, "
            "float, float]' = (0, 1), scaling: '_ScaleBase' = "
            "<manim.mobject.graphing.scale.LinearBase "
            "object>, dt: 'float' = 1e-08, discontinuities: "
            "'Iterable[float] | None' = None, use_smoothing: "
            "'bool' = True, use_vectorized: 'bool' = False, "
            "**kwargs: 'Any')",
        },
        "ParsableManimColor": {"kind": "constant", "module": "types", "signature": None},
        "PhaseFlow": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.movement",
            "signature": "(function: 'Callable[[np.ndarray], np.ndarray]', mobject: "
            "'Mobject', virtual_time: 'float' = 1, "
            "suspend_mobject_updating: 'bool' = False, rate_func: "
            "'RateFunction' = <function linear>, **kwargs: 'Any')",
        },
        "Point": {
            "bases": ["PMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.types.point_cloud_mobject",
            "signature": "(location: 'Point3DLike' = array([0., 0., 0.]), color: "
            "'ManimColor' = ManimColor('#000000'), **kwargs: 'Any') -> "
            "'None'",
        },
        "PointCloudDot": {
            "bases": ["Mobject1D", "PMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.types.point_cloud_mobject",
            "signature": "(center: 'Point3DLike' = array([0., 0., 0.]), radius: "
            "'float' = 2.0, stroke_width: 'int' = 2, density: "
            "'int' = 10, color: 'ManimColor' = "
            "ManimColor('#FFFF00'), **kwargs: 'Any') -> 'None'",
        },
        "PolarPlane": {
            "bases": ["Axes", "VGroup", "VMobject", "Mobject", "CoordinateSystem", "object"],
            "kind": "class",
            "module": "manim.mobject.graphing.coordinate_systems",
            "signature": "(radius_max: 'float' = 4.0, size: 'float | None' = None, "
            "radius_step: 'float' = 1, azimuth_step: 'float | None' = "
            "None, azimuth_units: 'str' = 'PI radians', "
            "azimuth_compact_fraction: 'bool' = True, azimuth_offset: "
            "'float' = 0, azimuth_direction: 'str' = 'CCW', "
            "azimuth_label_buff: 'float' = 0.1, "
            "azimuth_label_font_size: 'float' = 24, radius_config: "
            "'dict[str, Any] | None' = None, background_line_style: "
            "'dict[str, Any] | None' = None, faded_line_style: "
            "'dict[str, Any] | None' = None, faded_line_ratio: 'int' "
            "= 1, make_smooth_after_applying_functions: 'bool' = "
            "True, **kwargs: 'Any')",
        },
        "Polygon": {
            "bases": ["Polygram", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.polygram",
            "signature": "(*vertices: 'Point3DLike', **kwargs: 'Any') -> 'None'",
        },
        "Polygram": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.polygram",
            "signature": "(*vertex_groups: 'Point3DLike_Array', color: "
            "'ParsableManimColor' = ManimColor('#58C4DD'), **kwargs: "
            "'Any')",
        },
        "Polyhedron": {
            "bases": ["VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.polyhedra",
            "signature": "(vertex_coords: 'Point3DLike_Array', faces_list: "
            "'list[list[int]]', faces_config: 'dict[str, str | int | "
            "float | bool]' = {}, graph_config: 'dict[str, Any]' = "
            "{})",
        },
        "Prism": {
            "bases": ["Cube", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.three_dimensions",
            "signature": "(dimensions: 'Vector3DLike' = [3, 2, 1], **kwargs: 'Any') -> 'None'",
        },
        "QUALITIES": {"kind": "constant", "module": "", "signature": None},
        "R3_to_complex": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(point: 'Sequence[float]') -> 'np.ndarray'",
        },
        "RED": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "RED_A": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "RED_B": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "RED_C": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "RED_D": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "RED_E": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "RESAMPLING_ALGORITHMS": {"kind": "constant", "module": "", "signature": None},
        "RGBA": {
            "bases": ["object"],
            "kind": "class",
            "module": "manim.utils.color.core",
            "signature": "(value: 'ParsableManimColor | None', alpha: 'float' = 1.0) -> 'None'",
        },
        "RIGHT": {"kind": "constant", "module": "", "signature": None},
        "RandomColorGenerator": {
            "bases": ["object"],
            "kind": "class",
            "module": "manim.utils.color.core",
            "signature": "(seed: 'int | None' = None, sample_colors: "
            "'list[ManimColor] | None' = None) -> 'None'",
        },
        "Rectangle": {
            "bases": ["Polygon", "Polygram", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.polygram",
            "signature": "(color: 'ParsableManimColor' = ManimColor('#FFFFFF'), "
            "height: 'float' = 2.0, width: 'float' = 4.0, grid_xstep: "
            "'float | None' = None, grid_ystep: 'float | None' = None, "
            "mark_paths_closed: 'bool' = True, close_new_points: "
            "'bool' = True, **kwargs: 'Any')",
        },
        "RegularPolygon": {
            "bases": ["RegularPolygram", "Polygram", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.polygram",
            "signature": "(n: 'int' = 6, **kwargs: 'Any') -> 'None'",
        },
        "RegularPolygram": {
            "bases": ["Polygram", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.polygram",
            "signature": "(num_vertices: 'int', *, density: 'int' = 2, "
            "radius: 'float' = 1, start_angle: 'float | None' = "
            "None, **kwargs: 'Any') -> 'None'",
        },
        "RemoveTextLetterByLetter": {
            "bases": ["AddTextLetterByLetter", "ShowIncreasingSubsets", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.creation",
            "signature": "(text: 'Text', suspend_mobject_updating: "
            "'bool' = False, int_func: "
            "'Callable[[np.ndarray], np.ndarray]' = "
            "<ufunc 'ceil'>, rate_func: "
            "'Callable[[float], float]' = <function "
            "linear>, time_per_char: 'float' = 0.1, "
            "run_time: 'float | None' = None, "
            "reverse_rate_function=True, "
            "introducer=False, remover=True, **kwargs) "
            "-> 'None'",
        },
        "RendererType": {
            "bases": ["Enum", "object"],
            "kind": "class",
            "module": "manim.constants",
            "signature": "(*values)",
        },
        "ReplacementTransform": {
            "bases": ["Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(mobject: 'Mobject', target_mobject: 'Mobject', **kwargs) -> 'None'",
        },
        "Restore": {
            "bases": ["ApplyMethod", "Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(mobject: 'Mobject', **kwargs) -> 'None'",
        },
        "RightAngle": {
            "bases": ["Angle", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.line",
            "signature": "(line1: 'Line', line2: 'Line', length: 'float | None' = "
            "None, **kwargs: 'Any') -> 'None'",
        },
        "Rotate": {
            "bases": ["Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.rotation",
            "signature": "(mobject: 'Mobject', angle: 'float' = 3.141592653589793, "
            "axis: 'Vector3DLike' = array([0., 0., 1.]), about_point: "
            "'Point3DLike | None' = None, about_edge: 'Vector3DLike | "
            "None' = None, **kwargs: 'Any') -> 'None'",
        },
        "Rotating": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.rotation",
            "signature": "(mobject: 'Mobject', angle: 'float' = 6.283185307179586, "
            "axis: 'Vector3DLike' = array([0., 0., 1.]), about_point: "
            "'Point3DLike | None' = None, about_edge: 'Vector3DLike | "
            "None' = None, run_time: 'float' = 5, rate_func: "
            "'Callable[[float], float]' = <function linear>, **kwargs: "
            "'Any') -> 'None'",
        },
        "RoundedRectangle": {
            "bases": ["Rectangle", "Polygon", "Polygram", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.polygram",
            "signature": "(corner_radius: 'float | list[float]' = 0.5, **kwargs: 'Any')",
        },
        "SCALE_FACTOR_PER_FONT_POINT": {"kind": "constant", "module": "", "signature": None},
        "SCENE_NOT_FOUND_MESSAGE": {"kind": "constant", "module": "", "signature": None},
        "SEMIBOLD": {"kind": "constant", "module": "", "signature": None},
        "SEMILIGHT": {"kind": "constant", "module": "", "signature": None},
        "SHIFT_VALUE": {"kind": "constant", "module": "", "signature": None},
        "SMALL_BUFF": {"kind": "constant", "module": "", "signature": None},
        "START_X": {"kind": "constant", "module": "", "signature": None},
        "START_Y": {"kind": "constant", "module": "", "signature": None},
        "SVGMobject": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.svg.svg_mobject",
            "signature": "(file_name: 'str | os.PathLike | None' = None, "
            "should_center: 'bool' = True, height: 'float | None' = "
            "2, width: 'float | None' = None, color: "
            "'ParsableManimColor | None' = None, opacity: 'float | "
            "None' = None, fill_color: 'ParsableManimColor | None' = "
            "None, fill_opacity: 'float | None' = None, stroke_color: "
            "'ParsableManimColor | None' = None, stroke_opacity: "
            "'float | None' = None, stroke_width: 'float | None' = "
            "None, svg_default: 'dict | None' = None, "
            "path_string_config: 'dict | None' = None, use_svg_cache: "
            "'bool' = True, **kwargs: 'Any')",
        },
        "SampleSpace": {
            "bases": ["Rectangle", "Polygon", "Polygram", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.graphing.probability",
            "signature": "(height: 'float' = 3, width: 'float' = 3, fill_color: "
            "'ParsableManimColor' = ManimColor('#444444'), "
            "fill_opacity: 'float' = 1, stroke_width: 'float' = 0.5, "
            "stroke_color: 'ParsableManimColor' = "
            "ManimColor('#BBBBBB'), default_label_scale_val: 'float' "
            "= 1)",
        },
        "ScaleInPlace": {
            "bases": ["ApplyMethod", "Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(mobject: 'Mobject', scale_factor: 'float', **kwargs) -> 'None'",
        },
        "Scene": {
            "bases": ["object"],
            "kind": "class",
            "module": "manim.scene.scene",
            "signature": "(renderer: 'CairoRenderer | OpenGLRenderer | None' = None, "
            "camera_class: 'type[Camera]' = <class "
            "'manim.camera.camera.Camera'>, always_update_mobjects: 'bool' "
            "= False, random_seed: 'int | None' = None, skip_animations: "
            "'bool' = False) -> 'None'",
        },
        "SceneFileWriter": {
            "bases": ["object"],
            "kind": "class",
            "module": "manim.scene.scene_file_writer",
            "signature": "(renderer: 'CairoRenderer | OpenGLRenderer', "
            "scene_name: 'str', **kwargs: 'Any') -> 'None'",
        },
        "ScreenRectangle": {
            "bases": ["Rectangle", "Polygon", "Polygram", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.frame",
            "signature": "(aspect_ratio: 'float' = 1.7777777777777777, "
            "height: 'float' = 4, **kwargs: 'Any') -> 'None'",
        },
        "Section": {
            "bases": ["object"],
            "kind": "class",
            "module": "manim.scene.section",
            "signature": "(type_: 'str', video: 'str | None', name: 'str', "
            "skip_animations: 'bool') -> 'None'",
        },
        "Sector": {
            "bases": ["AnnularSector", "Arc", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(radius: 'float' = 1, **kwargs: 'Any') -> 'None'",
        },
        "ShowIncreasingSubsets": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.creation",
            "signature": "(group: 'Mobject', suspend_mobject_updating: "
            "'bool' = False, int_func: "
            "'Callable[[np.ndarray], np.ndarray]' = <ufunc "
            "'floor'>, reverse_rate_function=False, "
            "**kwargs) -> 'None'",
        },
        "ShowPartial": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.creation",
            "signature": "(mobject: 'VMobject | OpenGLVMobject | OpenGLSurface | None', **kwargs)",
        },
        "ShowPassingFlash": {
            "bases": ["ShowPartial", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.indication",
            "signature": "(mobject: 'VMobject', time_width: 'float' = 0.1, "
            "**kwargs: 'Any') -> 'None'",
        },
        "ShowPassingFlashWithThinningStrokeWidth": {
            "bases": ["AnimationGroup", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.indication",
            "signature": "(vmobject: 'VMobject', "
            "n_segments: 'int' = 10, "
            "time_width: 'float' = 0.1, "
            "remover: 'bool' = True, "
            "**kwargs: 'Any')",
        },
        "ShowSubmobjectsOneByOne": {
            "bases": ["ShowIncreasingSubsets", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.creation",
            "signature": "(group: 'Iterable[Mobject]', int_func: "
            "'Callable[[np.ndarray], np.ndarray]' = "
            "<ufunc 'ceil'>, **kwargs) -> 'None'",
        },
        "ShrinkToCenter": {
            "bases": ["ScaleInPlace", "ApplyMethod", "Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(mobject: 'Mobject', **kwargs) -> 'None'",
        },
        "SingleStringMathTex": {
            "bases": ["SVGMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.text.tex_mobject",
            "signature": "(tex_string: 'str', stroke_width: 'float' = 0, "
            "should_center: 'bool' = True, height: 'float | "
            "None' = None, organize_left_to_right: 'bool' = "
            "False, tex_environment: 'str | None' = "
            "'align*', tex_template: 'TexTemplate | None' = "
            "None, font_size: 'float' = 48, color: "
            "'ParsableManimColor | None' = None, **kwargs: "
            "'Any')",
        },
        "SmoothedVectorizedHomotopy": {
            "bases": ["Homotopy", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.movement",
            "signature": "(homotopy: 'Callable[[float, float, "
            "float, float], tuple[float, float, "
            "float]]', mobject: 'Mobject', run_time: "
            "'float' = 3, apply_function_kwargs: "
            "'dict[str, Any] | None' = None, "
            "**kwargs: 'Any')",
        },
        "SpecialThreeDScene": {
            "bases": ["ThreeDScene", "Scene", "object"],
            "kind": "class",
            "module": "manim.scene.three_d_scene",
            "signature": "(cut_axes_at_radius=True, "
            "camera_config={'should_apply_shading': True, "
            "'exponential_projection': True}, "
            "three_d_axes_config={'num_axis_pieces': 1, "
            "'axis_config': {'unit_size': 2, "
            "'tick_frequency': 1, "
            "'numbers_with_elongated_ticks': [0, 1, 2], "
            "'stroke_width': 2}}, sphere_config={'radius': 2, "
            "'resolution': (24, 48)}, "
            "default_angled_camera_position={'phi': "
            "1.2217304763960306, 'theta': "
            "-1.9198621771937625}, "
            "low_quality_config={'camera_config': "
            "{'should_apply_shading': False}, "
            "'three_d_axes_config': {'num_axis_pieces': 1}, "
            "'sphere_config': {'resolution': (12, 24)}}, "
            "**kwargs)",
        },
        "Sphere": {
            "bases": ["Surface", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.three_dimensions",
            "signature": "(center: 'Point3DLike' = array([0., 0., 0.]), radius: "
            "'float' = 1, resolution: 'int | Sequence[int] | None' = "
            "None, u_range: 'tuple[float, float]' = (0, "
            "6.283185307179586), v_range: 'tuple[float, float]' = (0, "
            "3.141592653589793), **kwargs: 'Any') -> 'None'",
        },
        "SpinInFromNothing": {
            "bases": ["GrowFromCenter", "GrowFromPoint", "Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.growing",
            "signature": "(mobject: 'Mobject', angle: 'float' = "
            "1.5707963267948966, point_color: "
            "'ParsableManimColor | None' = None, **kwargs: "
            "'Any')",
        },
        "SpiralIn": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.creation",
            "signature": "(shapes: 'Mobject', scale_factor: 'float' = 8, "
            "fade_in_fraction=0.3, **kwargs) -> 'None'",
        },
        "SplitScreenCamera": {
            "bases": ["OldMultiCamera", "Camera", "object"],
            "kind": "class",
            "module": "manim.camera.mapping_camera",
            "signature": "(left_camera, right_camera, **kwargs)",
        },
        "Square": {
            "bases": ["Rectangle", "Polygon", "Polygram", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.polygram",
            "signature": "(side_length: 'float' = 2.0, **kwargs: 'Any') -> 'None'",
        },
        "Star": {
            "bases": ["Polygon", "Polygram", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.polygram",
            "signature": "(n: 'int' = 5, *, outer_radius: 'float' = 1, inner_radius: "
            "'float | None' = None, density: 'int' = 2, start_angle: 'float "
            "| None' = 1.5707963267948966, **kwargs: 'Any') -> 'None'",
        },
        "StealthTip": {
            "bases": ["ArrowTip", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.tips",
            "signature": "(fill_opacity: 'float' = 1, stroke_width: 'float' = 3, "
            "length: 'float' = 0.175, start_angle: 'float' = "
            "3.141592653589793, **kwargs: 'Any')",
        },
        "StreamLines": {
            "bases": ["VectorField", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.vector_field",
            "signature": "(func: 'Callable[[np.ndarray], np.ndarray]', color: "
            "'ParsableManimColor | None' = None, color_scheme: "
            "'Callable[[np.ndarray], float] | None' = None, "
            "min_color_scheme_value: 'float' = 0, "
            "max_color_scheme_value: 'float' = 2, colors: "
            "'Sequence[ParsableManimColor]' = "
            "[ManimColor('#236B8E'), ManimColor('#83C167'), "
            "ManimColor('#F7D96F'), ManimColor('#FC6255')], x_range: "
            "'Sequence[float]' = None, y_range: 'Sequence[float]' = "
            "None, z_range: 'Sequence[float]' = None, "
            "three_dimensions: 'bool' = False, noise_factor: 'float "
            "| None' = None, n_repeats=1, dt=0.05, virtual_time=3, "
            "max_anchors_per_line=100, padding=3, stroke_width=1, "
            "opacity=1, **kwargs)",
        },
        "Succession": {
            "bases": ["AnimationGroup", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.composition",
            "signature": "(*animations: 'Animation', lag_ratio: 'float' = 1, **kwargs: 'Any')",
        },
        "Surface": {
            "bases": ["VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.three_dimensions",
            "signature": "(func: 'Callable[[float, float], np.ndarray]', u_range: "
            "'tuple[float, float]' = (0, 1), v_range: 'tuple[float, "
            "float]' = (0, 1), resolution: 'int | Sequence[int]' = 32, "
            "surface_piece_config: 'dict' = {}, fill_color: "
            "'ParsableManimColor' = ManimColor('#29ABCA'), fill_opacity: "
            "'float' = 1.0, checkerboard_colors: "
            "'Iterable[ParsableManimColor] | Literal[False]' = "
            "[ManimColor('#29ABCA'), ManimColor('#236B8E')], "
            "stroke_color: 'ParsableManimColor' = ManimColor('#BBBBBB'), "
            "stroke_width: 'float' = 0.5, should_make_jagged: 'bool' = "
            "False, pre_function_handle_to_anchor_scale_factor: 'float' "
            "= 1e-05, **kwargs: 'Any') -> 'None'",
        },
        "SurroundingRectangle": {
            "bases": [
                "RoundedRectangle",
                "Rectangle",
                "Polygon",
                "Polygram",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.geometry.shape_matchers",
            "signature": "(*mobjects: 'Mobject', color: "
            "'ParsableManimColor' = ManimColor('#FFFF00'), "
            "buff: 'float | tuple[float, float]' = 0.1, "
            "corner_radius: 'float' = 0.0, **kwargs: 'Any') "
            "-> 'None'",
        },
        "Swap": {
            "bases": ["CyclicReplace", "Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(*mobjects: 'Mobject', path_arc: 'float' = 1.5707963267948966, "
            "**kwargs) -> 'None'",
        },
        "TAU": {"kind": "constant", "module": "", "signature": None},
        "TEAL": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "TEAL_A": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "TEAL_B": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "TEAL_C": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "TEAL_D": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "TEAL_E": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "THIN": {"kind": "constant", "module": "", "signature": None},
        "Table": {
            "bases": ["VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.table",
            "signature": "(table: 'Iterable[Iterable[float | str | VMobject]]', "
            "row_labels: 'Iterable[VMobject] | None' = None, col_labels: "
            "'Iterable[VMobject] | None' = None, top_left_entry: 'VMobject "
            "| None' = None, v_buff: 'float' = 0.8, h_buff: 'float' = 1.3, "
            "include_outer_lines: 'bool' = False, "
            "add_background_rectangles_to_entries: 'bool' = False, "
            "entries_background_color: 'ParsableManimColor' = "
            "ManimColor('#000000'), include_background_rectangle: 'bool' = "
            "False, background_rectangle_color: 'ParsableManimColor' = "
            "ManimColor('#000000'), element_to_mobject: 'Callable[[float | "
            "str | VMobject], VMobject]' = <class "
            "'manim.mobject.text.text_mobject.Paragraph'>, "
            "element_to_mobject_config: 'dict' = {}, "
            "arrange_in_grid_config: 'dict' = {}, line_config: 'dict' = "
            "{}, **kwargs)",
        },
        "TangentLine": {
            "bases": ["Line", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.line",
            "signature": "(vmob: 'VMobject', alpha: 'float', length: 'float' = 1, "
            "d_alpha: 'float' = 1e-06, **kwargs: 'Any') -> 'None'",
        },
        "TangentialArc": {
            "bases": [
                "ArcBetweenPoints",
                "Arc",
                "TipableVMobject",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(line1: 'Line', line2: 'Line', radius: 'float', "
            "corner: 'Any' = (1, 1), **kwargs: 'Any')",
        },
        "Tetrahedron": {
            "bases": ["Polyhedron", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.polyhedra",
            "signature": "(edge_length: 'float' = 1, **kwargs: 'Any')",
        },
        "Tex": {
            "bases": [
                "MathTex",
                "SingleStringMathTex",
                "SVGMobject",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.text.tex_mobject",
            "signature": "(*tex_strings: 'str', arg_separator: 'str' = '', "
            "tex_environment: 'str | None' = 'center', **kwargs: 'Any')",
        },
        "TexFontTemplates": {
            "bases": ["object"],
            "kind": "class",
            "module": "manim.utils.tex_templates",
            "signature": "()",
        },
        "TexTemplate": {
            "bases": ["object"],
            "kind": "class",
            "module": "manim.utils.tex",
            "signature": "(tex_compiler: 'str' = 'latex', description: 'str' = "
            "'', output_format: 'str' = '.dvi', documentclass: 'str' "
            "= '\\\\documentclass[preview]{standalone}', preamble: "
            "'str' = "
            "'\\\\usepackage[english]{babel}\\n\\\\usepackage{amsmath}\\n\\\\usepackage{amssymb}', "
            "placeholder_text: 'str' = 'YourTextHere', "
            "post_doc_commands: 'str' = '') -> None",
        },
        "TexTemplateLibrary": {
            "bases": ["object"],
            "kind": "class",
            "module": "manim.utils.tex_templates",
            "signature": "()",
        },
        "Text": {
            "bases": ["SVGMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.text.text_mobject",
            "signature": "(text: 'str', fill_opacity: 'float' = 1.0, stroke_width: "
            "'float' = 0, color: 'ParsableManimColor | None' = None, "
            "font_size: 'float' = 48, line_spacing: 'float' = -1, font: "
            "'str' = '', slant: 'str' = 'NORMAL', weight: 'str' = 'NORMAL', "
            "t2c: 'dict[str, str] | None' = None, t2f: 'dict[str, str] | "
            "None' = None, t2g: 'dict[str, Iterable[ParsableManimColor]] | "
            "None' = None, t2s: 'dict[str, str] | None' = None, t2w: "
            "'dict[str, str] | None' = None, gradient: "
            "'Iterable[ParsableManimColor] | None' = None, tab_width: 'int' "
            "= 4, warn_missing_font: 'bool' = True, height: 'float | None' "
            "= None, width: 'float | None' = None, should_center: 'bool' = "
            "True, disable_ligatures: 'bool' = False, use_svg_cache: 'bool' "
            "= False, **kwargs: 'Any')",
        },
        "ThreeDAxes": {
            "bases": ["Axes", "VGroup", "VMobject", "Mobject", "CoordinateSystem", "object"],
            "kind": "class",
            "module": "manim.mobject.graphing.coordinate_systems",
            "signature": "(x_range: 'Sequence[float] | None' = (-6, 6, 1), "
            "y_range: 'Sequence[float] | None' = (-5, 5, 1), z_range: "
            "'Sequence[float] | None' = (-4, 4, 1), x_length: 'float "
            "| None' = 10.5, y_length: 'float | None' = 10.5, "
            "z_length: 'float | None' = 6.5, z_axis_config: "
            "'dict[str, Any] | None' = None, z_normal: 'Vector3DLike' "
            "= array([ 0., -1.,  0.]), num_axis_pieces: 'int' = 20, "
            "light_source: 'Point3DLike' = array([-7., -9., 10.]), "
            "depth: 'Any' = None, gloss: 'float' = 0.5, **kwargs: "
            "'dict[str, Any]')",
        },
        "ThreeDCamera": {
            "bases": ["Camera", "object"],
            "kind": "class",
            "module": "manim.camera.three_d_camera",
            "signature": "(focal_distance: 'float' = 20.0, shading_factor: "
            "'float' = 0.2, default_distance: 'float' = 5.0, "
            "light_source_start_point: 'Point3DLike' = array([-7., "
            "-9., 10.]), should_apply_shading: 'bool' = True, "
            "exponential_projection: 'bool' = False, phi: 'float' = "
            "0, theta: 'float' = -1.5707963267948966, gamma: "
            "'float' = 0, zoom: 'float' = 1, **kwargs: 'Any')",
        },
        "ThreeDScene": {
            "bases": ["Scene", "object"],
            "kind": "class",
            "module": "manim.scene.three_d_scene",
            "signature": "(camera_class=<class "
            "'manim.camera.three_d_camera.ThreeDCamera'>, "
            "ambient_camera_rotation=None, "
            "default_angled_camera_orientation_kwargs=None, "
            "**kwargs)",
        },
        "ThreeDVMobject": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.three_dimensions",
            "signature": "(shade_in_3d: 'bool' = True, **kwargs: 'Any')",
        },
        "TipableVMobject": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.arc",
            "signature": "(tip_length: 'float' = 0.35, normal_vector: "
            "'Vector3DLike' = array([0., 0., 1.]), tip_style: "
            "'dict | None' = None, **kwargs: 'Any') -> 'None'",
        },
        "Title": {
            "bases": [
                "Tex",
                "MathTex",
                "SingleStringMathTex",
                "SVGMobject",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.text.tex_mobject",
            "signature": "(*text_parts: 'str', include_underline: 'bool' = True, "
            "match_underline_width_to_text: 'bool' = False, "
            "underline_buff: 'float' = 0.25, **kwargs: 'Any')",
        },
        "Torus": {
            "bases": ["Surface", "VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.three_d.three_dimensions",
            "signature": "(major_radius: 'float' = 3, minor_radius: 'float' = 1, "
            "u_range: 'tuple[float, float]' = (0, 6.283185307179586), "
            "v_range: 'tuple[float, float]' = (0, 6.283185307179586), "
            "resolution: 'int | tuple[int, int] | None' = None, **kwargs: "
            "'Any') -> 'None'",
        },
        "TracedPath": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.animation.changing",
            "signature": "(traced_point_func: 'Callable', stroke_width: 'float' = "
            "2, stroke_color: 'ParsableManimColor | None' = "
            "ManimColor('#FFFFFF'), dissipating_time: 'float | None' "
            "= None, **kwargs: 'Any') -> 'None'",
        },
        "Transform": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(mobject: 'Mobject | None', target_mobject: 'Mobject | "
            "None' = None, path_func: 'Callable | None' = None, "
            "path_arc: 'float' = 0, path_arc_axis: 'np.ndarray' = "
            "array([0., 0., 1.]), path_arc_centers: 'Point3DLike | "
            "Point3DLike_Array | None' = None, "
            "replace_mobject_with_target_in_scene: 'bool' = False, "
            "**kwargs) -> 'None'",
        },
        "TransformAnimations": {
            "bases": ["Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(start_anim: 'Animation', end_anim: "
            "'Animation', rate_func: 'Callable' = <function "
            "squish_rate_func.<locals>.result>, **kwargs) -> "
            "'None'",
        },
        "TransformFromCopy": {
            "bases": ["Transform", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform",
            "signature": "(mobject: 'Mobject', target_mobject: 'Mobject', **kwargs) -> 'None'",
        },
        "TransformMatchingShapes": {
            "bases": ["TransformMatchingAbstractBase", "AnimationGroup", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform_matching_parts",
            "signature": "(mobject: 'Mobject', target_mobject: "
            "'Mobject', transform_mismatches: 'bool' = "
            "False, fade_transform_mismatches: 'bool' = "
            "False, key_map: 'dict | None' = None, "
            "**kwargs)",
        },
        "TransformMatchingTex": {
            "bases": ["TransformMatchingAbstractBase", "AnimationGroup", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.transform_matching_parts",
            "signature": "(mobject: 'Mobject', target_mobject: "
            "'Mobject', transform_mismatches: 'bool' = "
            "False, fade_transform_mismatches: 'bool' = "
            "False, key_map: 'dict | None' = None, "
            "**kwargs)",
        },
        "Triangle": {
            "bases": [
                "RegularPolygon",
                "RegularPolygram",
                "Polygram",
                "VMobject",
                "Mobject",
                "object",
            ],
            "kind": "class",
            "module": "manim.mobject.geometry.polygram",
            "signature": "(**kwargs: 'Any') -> 'None'",
        },
        "TrueDot": {
            "bases": ["DotCloud", "OpenGLPMobject", "OpenGLMobject", "object"],
            "kind": "class",
            "module": "manim.mobject.opengl.dot_cloud",
            "signature": "(center: 'Point3DLike' = array([0., 0., 0.]), stroke_width: "
            "'float' = 2.0, **kwargs: 'Any')",
        },
        "TypeWithCursor": {
            "bases": ["AddTextLetterByLetter", "ShowIncreasingSubsets", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.creation",
            "signature": "(text: 'Text', cursor: 'Mobject', buff: 'float' = "
            "0.1, keep_cursor_y: 'bool' = True, leave_cursor_on: "
            "'bool' = True, time_per_char: 'float' = 0.1, "
            "reverse_rate_function=False, introducer=True, "
            "**kwargs) -> 'None'",
        },
        "UL": {"kind": "constant", "module": "", "signature": None},
        "ULTRABOLD": {"kind": "constant", "module": "", "signature": None},
        "ULTRAHEAVY": {"kind": "constant", "module": "", "signature": None},
        "ULTRALIGHT": {"kind": "constant", "module": "", "signature": None},
        "UP": {"kind": "constant", "module": "", "signature": None},
        "UR": {"kind": "constant", "module": "", "signature": None},
        "Uncreate": {
            "bases": ["Create", "ShowPartial", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.creation",
            "signature": "(mobject: 'VMobject | OpenGLVMobject', "
            "reverse_rate_function: 'bool' = True, remover: 'bool' = "
            "True, **kwargs) -> 'None'",
        },
        "Underline": {
            "bases": ["Line", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.shape_matchers",
            "signature": "(mobject: 'Mobject', buff: 'float' = 0.1, **kwargs: 'Any') -> 'None'",
        },
        "Union": {
            "bases": ["_BooleanOps", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.boolean_ops",
            "signature": "(*vmobjects: 'VMobject', **kwargs: 'Any') -> 'None'",
        },
        "UnitInterval": {
            "bases": ["NumberLine", "Line", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.graphing.number_line",
            "signature": "(unit_size: 'float' = 10, "
            "numbers_with_elongated_ticks: 'list[float] | None' = "
            "None, decimal_number_config: 'dict[str, Any] | None' = "
            "None, **kwargs: 'Any')",
        },
        "UntypeWithCursor": {
            "bases": [
                "TypeWithCursor",
                "AddTextLetterByLetter",
                "ShowIncreasingSubsets",
                "Animation",
                "object",
            ],
            "kind": "class",
            "module": "manim.animation.creation",
            "signature": "(text: 'Text', cursor: 'VMobject | None' = None, "
            "time_per_char: 'float' = 0.1, "
            "reverse_rate_function=True, introducer=False, "
            "remover=True, **kwargs) -> 'None'",
        },
        "Unwrite": {
            "bases": ["Write", "DrawBorderThenFill", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.creation",
            "signature": "(vmobject: 'VMobject', rate_func: 'Callable[[float], "
            "float]' = <function linear>, reverse: 'bool' = True, "
            "**kwargs) -> 'None'",
        },
        "UpdateFromAlphaFunc": {
            "bases": ["UpdateFromFunc", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.updaters.update",
            "signature": "(mobject: 'Mobject', update_function: "
            "'Callable[[Mobject], Any]', "
            "suspend_mobject_updating: 'bool' = False, "
            "**kwargs: 'Any') -> 'None'",
        },
        "UpdateFromFunc": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.updaters.update",
            "signature": "(mobject: 'Mobject', update_function: "
            "'Callable[[Mobject], Any]', "
            "suspend_mobject_updating: 'bool' = False, **kwargs: "
            "'Any') -> 'None'",
        },
        "VDict": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.types.vectorized_mobject",
            "signature": "(mapping_or_iterable: 'Mapping[Hashable, VMobject] | "
            "Iterable[tuple[Hashable, VMobject]]' = {}, show_keys: 'bool' "
            "= False, **kwargs) -> 'None'",
        },
        "VGroup": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.types.vectorized_mobject",
            "signature": "(*vmobjects: 'VMobject | Iterable[VMobject]', **kwargs: 'Any') -> 'None'",
        },
        "VMobject": {
            "bases": ["Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.types.vectorized_mobject",
            "signature": "(fill_color: 'ParsableManimColor | None' = None, "
            "fill_opacity: 'float' = 0.0, stroke_color: "
            "'ParsableManimColor | None' = None, stroke_opacity: "
            "'float' = 1.0, stroke_width: 'float' = 4, "
            "background_stroke_color: 'ParsableManimColor | None' = "
            "ManimColor('#000000'), background_stroke_opacity: 'float' "
            "= 1.0, background_stroke_width: 'float' = 0, sheen_factor: "
            "'float' = 0.0, joint_type: 'LineJointType | None' = None, "
            "sheen_direction: 'Vector3DLike' = array([-1.,  1.,  0.]), "
            "close_new_points: 'bool' = False, "
            "pre_function_handle_to_anchor_scale_factor: 'float' = "
            "0.01, make_smooth_after_applying_functions: 'bool' = "
            "False, background_image: 'Image | str | None' = None, "
            "shade_in_3d: 'bool' = False, tolerance_for_point_equality: "
            "'float' = 1e-06, n_points_per_cubic_curve: 'int' = 4, "
            "cap_style: 'CapStyleType' = <CapStyleType.AUTO: 0>, "
            "**kwargs: 'Any')",
        },
        "VMobjectFromSVGPath": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.svg.svg_mobject",
            "signature": "(path_obj: 'se.Path', long_lines: 'bool' = "
            "False, should_subdivide_sharp_curves: 'bool' = "
            "False, should_remove_null_curves: 'bool' = "
            "False, **kwargs: 'Any')",
        },
        "ValueTracker": {
            "bases": ["Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.value_tracker",
            "signature": "(value: 'float' = 0, **kwargs: 'Any') -> 'None'",
        },
        "Variable": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.text.numbers",
            "signature": "(var: 'float', label: 'str | Tex | MathTex | Text | "
            "SingleStringMathTex', var_type: 'type[DecimalNumber | "
            "Integer]' = <class "
            "'manim.mobject.text.numbers.DecimalNumber'>, "
            "num_decimal_places: 'int' = 2, **kwargs: 'Any')",
        },
        "Vector": {
            "bases": ["Arrow", "Line", "TipableVMobject", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.geometry.line",
            "signature": "(direction: 'Vector2DLike | Vector3DLike' = array([1., 0., "
            "0.]), buff: 'float' = 0, **kwargs: 'Any') -> 'None'",
        },
        "VectorField": {
            "bases": ["VGroup", "VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.vector_field",
            "signature": "(func: 'Callable[[Point3D], Vector3D]', color: "
            "'ParsableManimColor | None' = None, color_scheme: "
            "'Callable[[Vector3D], float] | None' = None, "
            "min_color_scheme_value: 'float' = 0, "
            "max_color_scheme_value: 'float' = 2, colors: "
            "'Sequence[ParsableManimColor]' = "
            "[ManimColor('#236B8E'), ManimColor('#83C167'), "
            "ManimColor('#F7D96F'), ManimColor('#FC6255')], "
            "**kwargs)",
        },
        "VectorScene": {
            "bases": ["Scene", "object"],
            "kind": "class",
            "module": "manim.scene.vector_space_scene",
            "signature": "(basis_vector_stroke_width: 'float' = 6.0, **kwargs: 'Any') -> 'None'",
        },
        "VectorizedPoint": {
            "bases": ["VMobject", "Mobject", "object"],
            "kind": "class",
            "module": "manim.mobject.types.vectorized_mobject",
            "signature": "(location: 'Point3DLike' = array([0., 0., 0.]), "
            "color: 'ManimColor' = ManimColor('#000000'), "
            "fill_opacity: 'float' = 0, stroke_width: 'float' = "
            "0, artificial_width: 'float' = 0.01, "
            "artificial_height: 'float' = 0.01, **kwargs) -> "
            "'None'",
        },
        "WHITE": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "Wait": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.animation",
            "signature": "(run_time: 'float' = 1, stop_condition: 'Callable[[], bool] | "
            "None' = None, frozen_frame: 'bool | None' = None, rate_func: "
            "'Callable[[float], float]' = <function linear>, **kwargs)",
        },
        "Wiggle": {
            "bases": ["Animation", "object"],
            "kind": "class",
            "module": "manim.animation.indication",
            "signature": "(mobject: 'Mobject', scale_value: 'float' = 1.1, "
            "rotation_angle: 'float' = 0.06283185307179587, n_wiggles: "
            "'int' = 6, scale_about_point: 'Point3DLike | None' = None, "
            "rotate_about_point: 'Point3DLike | None' = None, run_time: "
            "'float' = 2, **kwargs: 'Any')",
        },
        "Write": {
            "bases": ["DrawBorderThenFill", "Animation", "object"],
            "kind": "class",
            "module": "manim.animation.creation",
            "signature": "(vmobject: 'VMobject | OpenGLVMobject', rate_func: "
            "'Callable[[float], float]' = <function linear>, reverse: "
            "'bool' = False, **kwargs) -> 'None'",
        },
        "X_AXIS": {"kind": "constant", "module": "", "signature": None},
        "YELLOW": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "YELLOW_A": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "YELLOW_B": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "YELLOW_C": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "YELLOW_D": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "YELLOW_E": {"kind": "constant", "module": "manim.utils.color.core", "signature": None},
        "Y_AXIS": {"kind": "constant", "module": "", "signature": None},
        "Z_AXIS": {"kind": "constant", "module": "", "signature": None},
        "ZoomedScene": {
            "bases": ["MovingCameraScene", "Scene", "object"],
            "kind": "class",
            "module": "manim.scene.zoomed_scene",
            "signature": "(camera_class: 'type[Camera]' = <class "
            "'manim.camera.multi_camera.MultiCamera'>, "
            "zoomed_display_height: 'float' = 3, "
            "zoomed_display_width: 'float' = 3, "
            "zoomed_display_center: 'Point3DLike | None' = None, "
            "zoomed_display_corner: 'Vector3D' = array([1., 1., "
            "0.]), zoomed_display_corner_buff: 'float' = 0.5, "
            "zoomed_camera_config: 'dict[str, Any]' = "
            "{'default_frame_stroke_width': 2, 'background_opacity': "
            "1}, zoomed_camera_image_mobject_config: 'dict[str, "
            "Any]' = {}, zoomed_camera_frame_starting_position: "
            "'Point3DLike' = array([0., 0., 0.]), zoom_factor: "
            "'float' = 0.15, image_frame_stroke_width: 'float' = 3, "
            "zoom_activated: 'bool' = False, **kwargs: 'Any') -> "
            "'None'",
        },
        "add_extension_if_not_present": {
            "kind": "callable",
            "module": "manim.utils.file_ops",
            "signature": "(file_name: 'Path', extension: 'str') -> 'Path'",
        },
        "adjacent_n_tuples": {
            "kind": "callable",
            "module": "manim.utils.iterables",
            "signature": "(objects: 'Sequence[T]', n: 'int') -> 'zip[tuple[T, ...]]'",
        },
        "adjacent_pairs": {
            "kind": "callable",
            "module": "manim.utils.iterables",
            "signature": "(objects: 'Sequence[T]') -> 'zip[tuple[T, ...]]'",
        },
        "all_elements_are_instances": {
            "kind": "callable",
            "module": "manim.utils.iterables",
            "signature": "(iterable: 'Iterable[object]', Class: 'type[object]') -> 'bool'",
        },
        "always": {
            "kind": "callable",
            "module": "manim.animation.updaters.mobject_update_utils",
            "signature": "(method: 'Callable', *args, **kwargs) -> 'Mobject'",
        },
        "always_redraw": {
            "kind": "callable",
            "module": "manim.animation.updaters.mobject_update_utils",
            "signature": "(func: 'Callable[[], Mobject]') -> 'Mobject'",
        },
        "always_rotate": {
            "kind": "callable",
            "module": "manim.animation.updaters.mobject_update_utils",
            "signature": "(mobject: 'Mobject', rate: 'float' = "
            "0.3490658503988659, **kwargs) -> 'Mobject'",
        },
        "always_shift": {
            "kind": "callable",
            "module": "manim.animation.updaters.mobject_update_utils",
            "signature": "(mobject: 'Mobject', direction: "
            "'np.ndarray[np.float64]' = array([1., 0., 0.]), rate: "
            "'float' = 0.1) -> 'Mobject'",
        },
        "angle_axis_from_quaternion": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(quaternion: 'Sequence[float]') -> 'Sequence[float]'",
        },
        "angle_between_vectors": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(v1: 'np.ndarray', v2: 'np.ndarray') -> 'float'",
        },
        "angle_of_vector": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(vector: 'Sequence[float] | np.ndarray') -> 'float'",
        },
        "annotations": {"kind": "constant", "module": "__future__", "signature": None},
        "assert_is_mobject_method": {
            "kind": "callable",
            "module": "manim.animation.updaters.mobject_update_utils",
            "signature": "(method: 'Callable') -> 'None'",
        },
        "average_color": {
            "kind": "callable",
            "module": "manim.utils.color.core",
            "signature": "(*colors: 'ParsableManimColor') -> 'ManimColor'",
        },
        "bezier": {
            "kind": "callable",
            "module": "manim.utils.bezier",
            "signature": "(points: 'Point3D_Array | Sequence[Point3D_Array]') -> "
            "'Callable[[float | ColVector], Point3D_Array]'",
        },
        "bezier_remap": {
            "kind": "callable",
            "module": "manim.utils.bezier",
            "signature": "(bezier_tuples: 'BezierPointsLike_Array', "
            "new_number_of_curves: 'int') -> 'BezierPoints_Array'",
        },
        "binary_search": {
            "kind": "callable",
            "module": "manim.utils.simple_functions",
            "signature": "(function: 'Callable[[float], float]', target: "
            "'float', lower_bound: 'float', upper_bound: 'float', "
            "tolerance: 'float' = 0.0001) -> 'float | None'",
        },
        "capture": {
            "kind": "callable",
            "module": "manim.utils.commands",
            "signature": "(command: 'str | list[str]', cwd: 'StrOrBytesPath | None' = "
            "None, command_input: 'str | None' = None) -> 'tuple[str, "
            "str, int]'",
        },
        "cartesian_to_spherical": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(vec: 'Vector3DLike') -> 'np.ndarray'",
        },
        "center_of_mass": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(points: 'PointNDLike_Array') -> 'PointND'",
        },
        "change_to_rgba_array": {
            "kind": "callable",
            "module": "manim.utils.images",
            "signature": "(image: 'RGBPixelArray', dtype: 'str' = 'uint8') -> 'RGBAPixelArray'",
        },
        "choose": {
            "kind": "callable",
            "module": "manim.utils.simple_functions",
            "signature": "(n: 'int', k: 'int') -> 'int'",
        },
        "cli_ctx_settings": {"kind": "constant", "module": "", "signature": None},
        "clip": {
            "kind": "callable",
            "module": "manim.utils.simple_functions",
            "signature": "(a: 'ComparableT', min_a: 'ComparableT', max_a: 'ComparableT') "
            "-> 'ComparableT'",
        },
        "clockwise_path": {
            "kind": "callable",
            "module": "manim.utils.paths",
            "signature": "() -> 'PathFuncType'",
        },
        "color_gradient": {
            "kind": "callable",
            "module": "manim.utils.color.core",
            "signature": "(reference_colors: 'Iterable[ParsableManimColor]', "
            "length_of_output: 'int') -> 'list[ManimColor]'",
        },
        "color_to_int_rgb": {
            "kind": "callable",
            "module": "manim.utils.color.core",
            "signature": "(color: 'ParsableManimColor') -> 'IntRGB'",
        },
        "color_to_int_rgba": {
            "kind": "callable",
            "module": "manim.utils.color.core",
            "signature": "(color: 'ParsableManimColor', alpha: 'float' = 1.0) -> 'IntRGBA'",
        },
        "color_to_rgb": {
            "kind": "callable",
            "module": "manim.utils.color.core",
            "signature": "(color: 'ParsableManimColor') -> 'FloatRGB'",
        },
        "color_to_rgba": {
            "kind": "callable",
            "module": "manim.utils.color.core",
            "signature": "(color: 'ParsableManimColor', alpha: 'float' = 1.0) -> 'FloatRGBA'",
        },
        "compass_directions": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(n: 'int' = 4, start_vect: 'np.ndarray' = "
            "array([1., 0., 0.])) -> 'np.ndarray'",
        },
        "complex_func_to_R3_func": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(complex_func: 'Callable[[complex], "
            "complex]') -> 'Callable[[Point3DLike], "
            "Point3D]'",
        },
        "complex_to_R3": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(complex_num: 'complex') -> 'np.ndarray'",
        },
        "concatenate_lists": {
            "kind": "callable",
            "module": "manim.utils.iterables",
            "signature": "(*list_of_lists: 'Iterable[T]') -> 'list[T]'",
        },
        "config": {"kind": "constant", "module": "manim._config.utils", "signature": None},
        "console": {"kind": "constant", "module": "rich.console", "signature": None},
        "counterclockwise_path": {
            "kind": "callable",
            "module": "manim.utils.paths",
            "signature": "() -> 'PathFuncType'",
        },
        "cross2d": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(a: 'Vector2D | Vector2D_Array', b: 'Vector2D | "
            "Vector2D_Array') -> 'ManimFloat | npt.NDArray[ManimFloat]'",
        },
        "cycle_animation": {
            "kind": "callable",
            "module": "manim.animation.updaters.mobject_update_utils",
            "signature": "(animation: 'Animation', **kwargs) -> 'Mobject'",
        },
        "double_smooth": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(t: 'float') -> 'float'",
        },
        "drag_pixels": {
            "kind": "callable",
            "module": "manim.utils.images",
            "signature": "(frames: 'Sequence[PixelArray]') -> 'list[np.ndarray]'",
        },
        "earclip_triangulation": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(verts: 'np.ndarray', ring_ends: 'list') -> 'list'",
        },
        "ensure_executable": {
            "kind": "callable",
            "module": "manim.utils.file_ops",
            "signature": "(path_to_exe: 'Path') -> 'bool'",
        },
        "error_console": {"kind": "constant", "module": "rich.console", "signature": None},
        "exponential_decay": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(t: 'float', half_life: 'float' = 0.1) -> 'float'",
        },
        "f_always": {
            "kind": "callable",
            "module": "manim.animation.updaters.mobject_update_utils",
            "signature": "(method: 'Callable[[Mobject], None]', *arg_generators, "
            "**kwargs) -> 'Mobject'",
        },
        "find_intersection": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(p0s: 'Point3DLike_Array', v0s: "
            "'Vector3DLike_Array', p1s: 'Point3DLike_Array', "
            "v1s: 'Vector3DLike_Array', threshold: 'float' = "
            "1e-05) -> 'list[Point3D]'",
        },
        "frame": {"kind": "constant", "module": "manim._config.utils", "signature": None},
        "get_3d_vmob_end_corner": {
            "kind": "callable",
            "module": "manim.mobject.three_d.three_d_utils",
            "signature": "(vmob: 'VMobject') -> 'Point3D'",
        },
        "get_3d_vmob_end_corner_index": {
            "kind": "callable",
            "module": "manim.mobject.three_d.three_d_utils",
            "signature": "(vmob: 'VMobject') -> 'int'",
        },
        "get_3d_vmob_end_corner_unit_normal": {
            "kind": "callable",
            "module": "manim.mobject.three_d.three_d_utils",
            "signature": "(vmob: 'VMobject') -> 'Vector3D'",
        },
        "get_3d_vmob_gradient_start_and_end_points": {
            "kind": "callable",
            "module": "manim.mobject.three_d.three_d_utils",
            "signature": "(vmob: 'VMobject') -> 'tuple[Point3D, Point3D]'",
        },
        "get_3d_vmob_start_corner": {
            "kind": "callable",
            "module": "manim.mobject.three_d.three_d_utils",
            "signature": "(vmob: 'VMobject') -> 'Point3D'",
        },
        "get_3d_vmob_start_corner_index": {
            "kind": "callable",
            "module": "manim.mobject.three_d.three_d_utils",
            "signature": "(vmob: 'VMobject') -> 'Literal[0]'",
        },
        "get_3d_vmob_start_corner_unit_normal": {
            "kind": "callable",
            "module": "manim.mobject.three_d.three_d_utils",
            "signature": "(vmob: 'VMobject') -> 'Vector3D'",
        },
        "get_3d_vmob_unit_normal": {
            "kind": "callable",
            "module": "manim.mobject.three_d.three_d_utils",
            "signature": "(vmob: 'VMobject', point_index: 'int') -> 'Vector3D'",
        },
        "get_det_text": {
            "kind": "callable",
            "module": "manim.mobject.matrix",
            "signature": "(matrix: 'Matrix', determinant: 'int | str | None' = "
            "None, background_rect: 'bool' = False, "
            "initial_scale_factor: 'float' = 2) -> 'VGroup'",
        },
        "get_dir_layout": {
            "kind": "callable",
            "module": "manim.utils.commands",
            "signature": "(dirpath: 'Path') -> 'Generator[str, None, None]'",
        },
        "get_full_raster_image_path": {
            "kind": "callable",
            "module": "manim.utils.images",
            "signature": "(image_file_name: 'str | PurePath') -> 'Path'",
        },
        "get_full_sound_file_path": {
            "kind": "callable",
            "module": "manim.utils.sounds",
            "signature": "(sound_file_name: 'StrPath') -> 'Path'",
        },
        "get_plugins": {
            "kind": "callable",
            "module": "manim.plugins.plugins_flags",
            "signature": "() -> 'dict[str, Any]'",
        },
        "get_shaded_rgb": {
            "kind": "callable",
            "module": "manim.utils.color.core",
            "signature": "(rgb: 'FloatRGB', point: 'Point3D', "
            "unit_normal_vect: 'Vector3D', light_source: "
            "'Point3D') -> 'FloatRGB'",
        },
        "get_smooth_cubic_bezier_handle_points": {
            "kind": "callable",
            "module": "manim.utils.bezier",
            "signature": "(anchors: 'Point3DLike_Array') -> 'tuple[Point3D_Array, Point3D_Array]'",
        },
        "get_unit_normal": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(v1: 'Vector3DLike', v2: 'Vector3DLike', tol: "
            "'float' = 1e-06) -> 'Vector3D'",
        },
        "get_video_metadata": {
            "kind": "callable",
            "module": "manim.utils.commands",
            "signature": "(path_to_video: 'str | os.PathLike') -> 'VideoMetadata'",
        },
        "get_winding_number": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(points: 'Sequence[np.ndarray]') -> 'float'",
        },
        "guarantee_empty_existence": {
            "kind": "callable",
            "module": "manim.utils.file_ops",
            "signature": "(path: 'Path') -> 'Path'",
        },
        "guarantee_existence": {
            "kind": "callable",
            "module": "manim.utils.file_ops",
            "signature": "(path: 'Path') -> 'Path'",
        },
        "hex_to_rgb": {
            "kind": "callable",
            "module": "manim.utils.color.core",
            "signature": "(hex_code: 'str') -> 'FloatRGB'",
        },
        "index_labels": {
            "kind": "callable",
            "module": "manim.utils.debug",
            "signature": "(mobject: 'Mobject', label_height: 'float' = 0.15, "
            "background_stroke_width: 'float' = 5, "
            "background_stroke_color: 'ManimColor' = "
            "ManimColor('#000000'), **kwargs: 'Any') -> 'VGroup'",
        },
        "integer_interpolate": {
            "kind": "callable",
            "module": "manim.utils.bezier",
            "signature": "(start: 'float', end: 'float', alpha: 'float') -> 'tuple[int, float]'",
        },
        "interpolate": {
            "kind": "callable",
            "module": "manim.utils.bezier",
            "signature": "(start: 'float | Point3D', end: 'float | Point3D', "
            "alpha: 'float | ColVector') -> 'float | ColVector | "
            "Point3D | Point3D_Array'",
        },
        "interpolate_color": {
            "kind": "callable",
            "module": "manim.utils.color.core",
            "signature": "(color1: 'ManimColorT', color2: 'ManimColorT', "
            "alpha: 'float') -> 'ManimColorT'",
        },
        "inverse_interpolate": {
            "kind": "callable",
            "module": "manim.utils.bezier",
            "signature": "(start: 'float | Point3D', end: 'float | "
            "Point3D', value: 'float | Point3D') -> 'float | "
            "Point3D'",
        },
        "invert_color": {
            "kind": "callable",
            "module": "manim.utils.color.core",
            "signature": "(color: 'ManimColorT') -> 'ManimColorT'",
        },
        "invert_image": {
            "kind": "callable",
            "module": "manim.utils.images",
            "signature": "(image: 'PixelArray') -> 'Image'",
        },
        "is_closed": {
            "kind": "callable",
            "module": "manim.utils.bezier",
            "signature": "(points: 'Point3D_Array') -> 'bool'",
        },
        "is_gif_format": {
            "kind": "callable",
            "module": "manim.utils.file_ops",
            "signature": "() -> 'bool'",
        },
        "is_mov_format": {
            "kind": "callable",
            "module": "manim.utils.file_ops",
            "signature": "() -> 'bool'",
        },
        "is_mp4_format": {
            "kind": "callable",
            "module": "manim.utils.file_ops",
            "signature": "() -> 'bool'",
        },
        "is_png_format": {
            "kind": "callable",
            "module": "manim.utils.file_ops",
            "signature": "() -> 'bool'",
        },
        "is_webm_format": {
            "kind": "callable",
            "module": "manim.utils.file_ops",
            "signature": "() -> 'bool'",
        },
        "line_intersection": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(line1: 'Sequence[np.ndarray]', line2: "
            "'Sequence[np.ndarray]') -> 'np.ndarray'",
        },
        "linear": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(t: 'float') -> 'float'",
        },
        "lingering": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(t: 'float') -> 'float'",
        },
        "list_difference_update": {
            "kind": "callable",
            "module": "manim.utils.iterables",
            "signature": "(l1: 'Iterable[T]', l2: 'Iterable[T]') -> 'list[T]'",
        },
        "list_plugins": {
            "kind": "callable",
            "module": "manim.plugins.plugins_flags",
            "signature": "() -> 'None'",
        },
        "list_update": {
            "kind": "callable",
            "module": "manim.utils.iterables",
            "signature": "(l1: 'Iterable[T]', l2: 'Iterable[T]') -> 'list[T]'",
        },
        "listify": {
            "kind": "callable",
            "module": "manim.utils.iterables",
            "signature": "(obj: 'str | Iterable[T] | T') -> 'list[str] | list[T]'",
        },
        "logger": {"kind": "constant", "module": "logging", "signature": None},
        "make_even": {
            "kind": "callable",
            "module": "manim.utils.iterables",
            "signature": "(iterable_1: 'Iterable[T]', iterable_2: 'Iterable[U]') -> "
            "'tuple[list[T], list[U]]'",
        },
        "make_even_by_cycling": {
            "kind": "callable",
            "module": "manim.utils.iterables",
            "signature": "(iterable_1: 'Collection[T]', iterable_2: "
            "'Collection[U]') -> 'tuple[list[T], list[U]]'",
        },
        "match_interpolate": {
            "kind": "callable",
            "module": "manim.utils.bezier",
            "signature": "(new_start: 'float', new_end: 'float', old_start: "
            "'float', old_end: 'float', old_value: 'float | "
            "Point3D') -> 'float | Point3D'",
        },
        "matrix_to_mobject": {
            "kind": "callable",
            "module": "manim.mobject.matrix",
            "signature": "(matrix: 'np.ndarray') -> 'MathTex'",
        },
        "matrix_to_tex_string": {
            "kind": "callable",
            "module": "manim.mobject.matrix",
            "signature": "(matrix: 'np.ndarray') -> 'str'",
        },
        "merge_dicts_recursively": {
            "kind": "callable",
            "module": "manim.utils.config_ops",
            "signature": "(*dicts: 'dict[Any, Any]') -> 'dict[Any, Any]'",
        },
        "mid": {
            "kind": "callable",
            "module": "manim.utils.bezier",
            "signature": "(start: 'float | Point3D', end: 'float | Point3D') -> 'float | Point3D'",
        },
        "midpoint": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(point1: 'Sequence[float]', point2: 'Sequence[float]') -> "
            "'float | np.ndarray'",
        },
        "modify_atime": {
            "kind": "callable",
            "module": "manim.utils.file_ops",
            "signature": "(file_path: 'str') -> 'None'",
        },
        "normalize": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(vect: 'np.ndarray | tuple[float]', fall_back: "
            "'np.ndarray | None' = None) -> 'np.ndarray'",
        },
        "not_quite_there": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(func: 'RateFunction' = <function smooth>, "
            "proportion: 'float' = 0.7) -> 'RateFunction'",
        },
        "open_file": {
            "kind": "callable",
            "module": "manim.utils.file_ops",
            "signature": "(file_path: 'Path', in_browser: 'bool' = False) -> 'None'",
        },
        "override_animate": {
            "kind": "callable",
            "module": "manim.mobject.mobject",
            "signature": "(method) -> 'types.FunctionType'",
        },
        "override_animation": {
            "kind": "callable",
            "module": "manim.animation.animation",
            "signature": "(animation_class: 'type[Animation]') -> 'Callable[[Callable], Callable]'",
        },
        "partial_bezier_points": {
            "kind": "callable",
            "module": "manim.utils.bezier",
            "signature": "(points: 'BezierPointsLike', a: 'float', b: 'float') -> 'BezierPoints'",
        },
        "path_along_arc": {
            "kind": "callable",
            "module": "manim.utils.paths",
            "signature": "(arc_angle: 'float', axis: 'Vector3DLike' = "
            "array([0., 0., 1.])) -> 'PathFuncType'",
        },
        "perpendicular_bisector": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(line: 'Sequence[np.ndarray]', norm_vector: "
            "'Vector3D' = array([0., 0., 1.])) -> "
            "'Sequence[np.ndarray]'",
        },
        "point_lies_on_bezier": {
            "kind": "callable",
            "module": "manim.utils.bezier",
            "signature": "(point: 'Point3DLike', control_points: "
            "'BezierPointsLike', round_to: 'float' = 1e-06) "
            "-> 'bool'",
        },
        "print_family": {
            "kind": "callable",
            "module": "manim.utils.debug",
            "signature": "(mobject: 'Mobject', n_tabs: 'int' = 0) -> 'None'",
        },
        "proportions_along_bezier_curve_for_point": {
            "kind": "callable",
            "module": "manim.utils.bezier",
            "signature": "(point: 'Point3DLike', "
            "control_points: "
            "'BezierPointsLike', "
            "round_to: 'float' = 1e-06) "
            "-> 'MatrixMN'",
        },
        "quaternion_conjugate": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(quaternion: 'Sequence[float]') -> 'np.ndarray'",
        },
        "quaternion_from_angle_axis": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(angle: 'float', axis: 'np.ndarray', "
            "axis_normalized: 'bool' = False) -> "
            "'list[float]'",
        },
        "quaternion_mult": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(*quats: 'Sequence[float]') -> 'np.ndarray | list[float | np.ndarray]'",
        },
        "random_bright_color": {
            "kind": "callable",
            "module": "manim.utils.color.core",
            "signature": "() -> 'ManimColor'",
        },
        "random_color": {
            "kind": "callable",
            "module": "manim.utils.color.core",
            "signature": "() -> 'ManimColor'",
        },
        "register_font": {
            "kind": "callable",
            "module": "manim.mobject.text.text_mobject",
            "signature": "(font_file: 'str | Path') -> 'Iterator[None]'",
        },
        "regular_vertices": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(n: 'int', *, radius: 'float' = 1, start_angle: "
            "'float | None' = None) -> 'tuple[np.ndarray, "
            "float]'",
        },
        "remove_list_redundancies": {
            "kind": "callable",
            "module": "manim.utils.iterables",
            "signature": "(lst: 'Reversible[H]') -> 'list[H]'",
        },
        "remove_nones": {
            "kind": "callable",
            "module": "manim.utils.iterables",
            "signature": "(sequence: 'Iterable[T | None]') -> 'list[T]'",
        },
        "rgb_to_color": {
            "kind": "callable",
            "module": "manim.utils.color.core",
            "signature": "(rgb: 'FloatRGBLike | IntRGBLike') -> 'ManimColor'",
        },
        "rgb_to_hex": {
            "kind": "callable",
            "module": "manim.utils.color.core",
            "signature": "(rgb: 'FloatRGBLike | IntRGBLike') -> 'str'",
        },
        "rgba_to_color": {
            "kind": "callable",
            "module": "manim.utils.color.core",
            "signature": "(rgba: 'FloatRGBALike | IntRGBALike') -> 'ManimColor'",
        },
        "rotate_vector": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(vector: 'Vector3DLike', angle: 'float', axis: "
            "'Vector3DLike' = array([0., 0., 1.])) -> 'Vector3D'",
        },
        "rotation_about_z": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(angle: 'float') -> 'np.ndarray'",
        },
        "rotation_matrix": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(angle: 'float', axis: 'Vector3DLike', homogeneous: "
            "'bool' = False) -> 'np.ndarray'",
        },
        "running_start": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(t: 'float', pull_factor: 'float' = -0.5) -> 'float'",
        },
        "rush_from": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(t: 'float', inflection: 'float' = 10.0) -> 'float'",
        },
        "rush_into": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(t: 'float', inflection: 'float' = 10.0) -> 'float'",
        },
        "seek_full_path_from_defaults": {
            "kind": "callable",
            "module": "manim.utils.file_ops",
            "signature": "(file_name: 'StrPath', default_dir: "
            "'Path', extensions: 'list[str]') -> "
            "'Path'",
        },
        "shoelace": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(x_y: 'Point2D_Array') -> 'float'",
        },
        "shoelace_direction": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(x_y: 'Point2D_Array') -> 'str'",
        },
        "sigmoid": {
            "kind": "callable",
            "module": "manim.utils.simple_functions",
            "signature": "(x: 'float') -> 'float'",
        },
        "slow_into": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(t: 'float') -> 'float'",
        },
        "smooth": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(t: 'float', inflection: 'float' = 10.0) -> 'float'",
        },
        "smoothererstep": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(t: 'float') -> 'float'",
        },
        "smootherstep": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(t: 'float') -> 'float'",
        },
        "smoothstep": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(t: 'float') -> 'float'",
        },
        "spherical_to_cartesian": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(spherical: 'Sequence[float]') -> 'np.ndarray'",
        },
        "split_bezier": {
            "kind": "callable",
            "module": "manim.utils.bezier",
            "signature": "(points: 'BezierPointsLike', t: 'float') -> 'Spline'",
        },
        "squish_rate_func": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(func: 'RateFunction', a: 'float' = 0.4, b: "
            "'float' = 0.6) -> 'RateFunction'",
        },
        "straight_path": {
            "kind": "callable",
            "module": "manim.utils.paths",
            "signature": "() -> 'PathFuncType'",
        },
        "stretch_array_to_length": {
            "kind": "callable",
            "module": "manim.utils.iterables",
            "signature": "(nparray: 'npt.NDArray[F]', length: 'int') -> 'npt.NDArray[F]'",
        },
        "subdivide_bezier": {
            "kind": "callable",
            "module": "manim.utils.bezier",
            "signature": "(points: 'BezierPointsLike', n_divisions: 'int') -> 'Spline'",
        },
        "tempconfig": {
            "kind": "callable",
            "module": "manim._config",
            "signature": "(temp: 'ManimConfig | dict[str, Any]') -> 'Generator[None, None, None]'",
        },
        "there_and_back": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(t: 'float', inflection: 'float' = 10.0) -> 'float'",
        },
        "there_and_back_with_pause": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(t: 'float', pause_ratio: 'float' = 0.3333333333333333) -> 'float'",
        },
        "thick_diagonal": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(dim: 'int', thickness: 'int' = 2) -> 'MatrixMN'",
        },
        "tuplify": {
            "kind": "callable",
            "module": "manim.utils.iterables",
            "signature": "(obj: 'str | Iterable[T] | T') -> 'tuple[str] | tuple[T]'",
        },
        "turn_animation_into_updater": {
            "kind": "callable",
            "module": "manim.animation.updaters.mobject_update_utils",
            "signature": "(animation: 'Animation', cycle: 'bool' "
            "= False, delay: 'float' = 0, **kwargs) "
            "-> 'Mobject'",
        },
        "update_dict_recursively": {
            "kind": "callable",
            "module": "manim.utils.config_ops",
            "signature": "(current_dict: 'dict[Any, Any]', *others: 'dict[Any, Any]') -> 'None'",
        },
        "wiggle": {
            "kind": "callable",
            "module": "manim.utils.rate_functions",
            "signature": "(t: 'float', wiggles: 'float' = 2) -> 'float'",
        },
        "write_to_movie": {
            "kind": "callable",
            "module": "manim.utils.file_ops",
            "signature": "() -> 'bool'",
        },
        "z_to_vector": {
            "kind": "callable",
            "module": "manim.utils.space_ops",
            "signature": "(vector: 'np.ndarray') -> 'np.ndarray'",
        },
    },
    "version": "0.20.1",
}
