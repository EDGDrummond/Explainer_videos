#Manim Community v0.14.0

import inspect

from manim import *
from manim.utils.rate_functions import ease_out_bounce, ease_in_bounce, ease_out_sine, there_and_back, ease_in_out_sine
from manim_physics import *

# Adapted from the Charge class in manim-physics
class DotGlow(VGroup):

    def __init__(self, magnitude=1, point=ORIGIN, color1=RED_D, color2=RED_A, **kwargs):
        VGroup.__init__(self, **kwargs)

        layer_num = 80
        color_list = color_gradient([color1, color2], layer_num)
        opacity_func = lambda t: 1500 * (1 - abs(t - 0.009) ** 0.0001)
        rate_func = lambda t: t ** 2

        for i in range(layer_num):
            self.add(
                Arc(
                    radius=magnitude * rate_func((0.5 + i) / layer_num),
                    angle=TAU,
                    color=color_list[i],
                    stroke_width=101
                    * (rate_func((i + 1) / layer_num) - rate_func(i / layer_num))
                    * magnitude,
                    stroke_opacity=opacity_func(rate_func(i / layer_num)),
                ).shift(point)
            )

class Glow(VGroup):
    def __init__(self, mobject, magnitude=1, color1=RED_D, color2=RED_A, **kwargs):
        VGroup.__init__(self, **kwargs)

        center = mobject.get_center()
        layer_num = 80
        color_list = color_gradient([color1, color2], layer_num)
        opacity_func = lambda t: 1500 * (1 - abs(t - 0.009) ** 0.0001)
        rate_func = lambda t: t ** 2

        for i in range(layer_num):
            self.add(
                mobject.copy()
                # .scale(magnitude * i)
                .set_color(color_list[i])
                .set_stroke(
                    width=101 * (rate_func((i + 1) / layer_num) - rate_func(i / layer_num)) * magnitude,
                    opacity=opacity_func(rate_func(i / layer_num)))
                .shift(center)
            )

from shapely import geometry as gm
def intersection(vmob1: VMobject, vmob2: VMobject):
    """intersection points of 2 curves"""
    a = gm.LineString(vmob1.points)
    b = gm.LineString(vmob2.points)
    intersects: gm.GeometryCollection = a.intersection(b)
    try:  # for intersections > 1
        return np.array(
            [[[x, y, z] for x, y, z in m.coords][0] for m in intersects.geoms]
        )
    except:  # else
        return np.array([[x, y, z] for x, y, z in intersects.coords])

class Video(MovingCameraScene):
    def construct(self):
        BG = "#12141d"
        DARK = "#154bf9"
        LIGHT = "#00c0f9"
        RED = "#ff6f5f"
        PINK = "#e561e5"
        PURPLE = "#931cff"
        BLACK = "#000000"
        WHITE = "#ffffff"
        
        self.camera.background_color = BLACK
        # self.camera.frame.scale(4)

        axes = Axes([-6, 10], [-8, 8]).set_color(WHITE)
        
        self.play(FadeIn(axes))
        self.wait(1.5)
        
        # axes.y_axis.submobjects[0].set_y(axes.y_axis.submobjects[0].get_bottom()[1] - axes.y_axis.submobjects[1][0].get_top()[1])
        # print(axes.y_axis.submobjects)
        
        EC_generic_text = MathTex("y^{2} = x^{3} + Ax + B").to_corner(UR)
        curve_text = MathTex("y^{2} = x^{3} - 3x + 3").next_to(EC_generic_text, DOWN)
        underline = Line(curve_text.get_left(), curve_text.get_right(), color=LIGHT, stroke_width=0.75).next_to(curve_text, DOWN, buff=SMALL_BUFF)

        graph = axes.plot_implicit_curve(
            lambda x, y: x**3 - 3*x + 3 - y**2,
            color=PURPLE
        )
        # self.add(Glow(graph, 0.3))
        # self.add(graph)
        self.play(Write(EC_generic_text, run_time = 1.5))
        self.wait(1.5)
        self.play(
            # FadeIn(graph),#, shift=5*RIGHT, rate_func=ease_out_sine, run_time=0.75),
            Write(curve_text, run_time=1.5),
            # GrowFromEdge(underline, LEFT, point_color=WHITE)
            # Write(graph)
        )
        self.play(
            Circumscribe(curve_text, color=LIGHT)
            # GrowFromEdge(underline, LEFT, rate_func=rush_from)
        )
        self.play(
            FadeIn(graph)
        )
        # self.make_static_body(graph)#Line(6*LEFT, 6*RIGHT).shift(4*DOWN))
        self.wait(1.5)
        self.play(FadeOut(EC_generic_text))
        self.wait(1.5)
        self.play(curve_text.animate.shift(UP), run_time = 1)
        self.wait(1.5)

        coord1 = [-2, 1]
        coord2 = [-1.29, 2.17, 0]
        R = Dot(point=axes.c2p(*coord1), color=WHITE)
        # test_square = Square()
        R_glow = DotGlow(magnitude=0.3, point=R.get_center(), color1=LIGHT, color2=DARK)
        R_glow_weak = DotGlow(magnitude=0.1, point=R.get_center(), color1=LIGHT, color2=DARK)
        # R_glow.add_updater(
        #     lambda g: g.move_to(R.get_center())
        # )

        R2 = Dot(point=axes.c2p(*coord2), color=WHITE)
        R2_glow_weak = DotGlow(magnitude=0.1, point=R2.get_center(), color1=LIGHT, color2=DARK)
        R2_glow = DotGlow(magnitude=0.3, point=R2.get_center(), color1=LIGHT, color2=DARK)
        
        axes.add_coordinates([coord1[0], coord1[0]], [coord1[1], coord1[1]])
        labels = axes.coordinate_labels
        # axes.remove(labels)
        # self.wait(1.5)
        
        bg_recs = VGroup(*[BackgroundRectangle(label, color=BG, buff=0.05) for label in labels]).set_z_index(10)
        axes.coordinate_labels.set_z_index(11)

        # self.add(R)#, test_square)
        # self.make_rigid_body(R)
        self.play(
            FadeIn(R),
            # FadeIn(labels),
            FadeIn(bg_recs),
            FadeIn(R_glow)
        )

        # self.interactive_embed()
        self.wait(1.5)

        axes.get_x_axis().remove(axes.get_x_axis().submobjects[-1])
        axes.get_y_axis().remove(axes.get_y_axis().submobjects[-1])
        self.remove(bg_recs)

        self.play(
            FadeOut(R, rate_func=ease_out_bounce),
            Transform(R_glow, R_glow_weak, rate_func=ease_out_bounce, remover=True)
            # MoveAlongPath(R, graph)
        )
        # self.play(
        #     FadeIn(R2, rate_func=ease_in_bounce),
        #     ReplacementTransform(R2_glow_weak, R2_glow, rate_func=ease_in_bounce)
        # )
        # self.remove(R2_glow)
        # self.camera.frame.move_to(R2).scale(0.3)
        tangent = TangentLine(graph, 0.59, length=9).move_to(R2)#Line(LEFT, RIGHT)

        # tangent.rotate(angle_of_tangent(coord2[0], coord2[1], -1.28, 2.178)).move_to(R2.get_center())
        # print(axes.slope_of_tangent(-1.29, graph))

        # for x in [x/10000 for x in range(5000, 10000)]:
        #     print(x, graph.point_from_proportion(x), graph.proportion_from_point(graph.point_from_proportion(x)))

        # Adding a point to itself
        P_inter = Dot(intersection(graph, tangent)[-1])
        Ptimes2 = P_inter.copy().set_y(-P_inter.get_y())
        self_add = MathTex("P + P = 2P").next_to(EC_generic_text, DOWN)

        self.play(FadeIn(R2), Write(self_add, run_time = 1.5))
        self.play(FadeIn(tangent))
        self.wait(1.5)
        self.play(FadeIn(P_inter))
        self.wait(1.5)
        self.play(FadeOut(tangent))

        # P_2P = Line(P_inter.copy().set_y(P_inter.get_y()+2), Ptimes2.copy().set_y(Ptimes2.get_y()-2))
        Pinter_2P = Line(P_inter, Ptimes2.copy().set_y(Ptimes2.get_y())).set_length(6)
        self.play(FadeIn(Pinter_2P))
        self.wait(1.5)
        self.play(FadeIn(Ptimes2))
        self.wait(1.5)
        self.play(FadeOut(Pinter_2P))
        self.wait(1.5)
        self.play(FadeOut(P_inter))
        self.wait(1.5)
        self.play(FadeOut(self_add))

        # Adding different points
        P_2P = Line(R2, Ptimes2.copy().set_y(Ptimes2.get_y())).set_length(7)
        diff_add = MathTex("P + 2P = 3P").next_to(EC_generic_text, DOWN)

        self.play(FadeIn(P_2P), Write(diff_add, run_time = 1.5))
        self.wait(1.5)

        P3_inter = Dot(intersection(graph, P_2P)[1])

        self.play(FadeIn(P3_inter))
        self.wait(1.5)
        self.play(FadeOut(P_2P))
        self.wait(1.5)

        P3 = P3_inter.copy().set_y(-P3_inter.get_y())
        P3inter_3P = Line(P3_inter, P3).set_length(6)
        self.play(FadeIn(P3inter_3P))
        self.wait(1.5)
        self.play(FadeIn(P3))
        self.wait(1.5)
        self.play(FadeOut(P3inter_3P), FadeOut(P3_inter))
        self.wait(1.5)
        self.play(FadeOut(P3), FadeOut(Ptimes2), FadeOut(diff_add))
        self.wait(1.5)
        
        # Identity point
        identity_equ = MathTex("P + I = P").next_to(EC_generic_text, DOWN)
        self.play(Write(identity_equ, run_time = 1.5))
        self.wait(1.5)

        minus_P = R2.copy().set_y(-R2.get_y())
        self.play(FadeIn(minus_P))
        self.wait(1.5)
        P_ident = Line(R2, minus_P).set_length(8)
        self.play(FadeIn(P_ident))


class Infinity(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BLACK
        # self.camera.frame.scale(4)

        axes = Axes([-4, 4], [-4, 4], x_length = 4, y_length = 4, tips = True).set_color(WHITE)
        graph = axes.plot_implicit_curve(
            lambda x, y: x**3 - 3*x + 3 - y**2,
            color=PURPLE
        )
        
        self.play(
            # FadeIn(graph),
            FadeIn(axes),
        )
        self.wait(1.5)

        circle = Circle(radius=3, color=WHITE)
        self.play(FadeIn(circle))
        self.wait(1.5)



        
class Computation(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        LIGHT = "#00c0f9"
        DARK = "#154bf9"
        colour_special = "#931CFF"
        colour_special_darker = "#9C7900"
        colours_1 = ["#FFCC17", "#FF5555", "#E561E5", "#FF7D54"]
        colours_2 = ["#A479E0", "#4DECAA", "#FFD954", "#22F1E4"]

        # Introduce the equation we are computing, and the separate steps
        equation = MathTex("x^{3} + x + 5 = 35", color=colour_special)
        steps = MathTex(r'{{\operatorname{int} &= x*x}} \\ {{\operatorname{int}_2 &= \operatorname{int}*x}} \\ {{\operatorname{int}_3 &= \operatorname{int}_2 + x}} \\ {{\operatorname{out} &= \operatorname{int}_3 + 5}}')
        ops = MathTex(r'y &= x \\ y &= x \operatorname{(op)} z')
        vector_compact_1_square = Matrix([["a_1", "b_1", "c_1"]],left_bracket="(", right_bracket=")")
        vector_compact_1 = MathTex("(a_1, b_1, c_1)").shift(5.2*RIGHT + 2*UP)
        vector_compact_2 = MathTex("(a_2, b_2, c_2)")
        vector_compact_3 = MathTex("(a_3, b_3, c_3)")
        vector_compact_4 = MathTex("(a_4, b_4, c_4)").shift(5.2*RIGHT + 1.2*DOWN)
        vector_compact = Group(vector_compact_1, vector_compact_2, vector_compact_3, vector_compact_4).arrange(DOWN).move_to(5*RIGHT)
        vector_compact_i = MathTex("(a_i, b_i, c_i)")
        vector_dots = MathTex("\\vdots").shift(5.2*RIGHT + DOWN)
        vector_compact_n = MathTex("(a_n, b_n, c_n)")
        A = Matrix([["a_{11}"], ["\\vdots"], ["a_{1m}"]])
        B = Matrix([["b_{11}"], ["\\vdots"], ["b_{1m}"]])
        C = Matrix([["c_{11}"], ["\\vdots"], ["c_{1m}"]])
        vectors = Group(A, B, C).arrange()
        items = MathTex("{{(}} {{1,\hspace{2mm}}} {{x,\hspace{2mm}}} {{\operatorname{int},\hspace{2mm}}} {{\operatorname{int}_2,\hspace{2mm}}} {{\operatorname{int}_3,\hspace{2mm}}} {{\operatorname{out}}} {{)}}").shift(3.5*DOWN)
        AA = Matrix([["a_{11}"], ["a_{12}"], ["a_{13}"], ["a_{14}"], ["a_{15}"], ["a_{16}"]])
        BB = Matrix([["b_{11}"], ["b_{12}"], ["b_{13}"], ["b_{14}"], ["b_{15}"], ["b_{16}"]])
        CC = Matrix([["c_{11}"], ["c_{12}"], ["c_{13}"], ["c_{14}"], ["c_{15}"], ["c_{16}"]])
        vectorss = Group(AA, BB, CC).arrange()
        S = Matrix([[1], ["x"], ["\operatorname{int}"], ["\operatorname{int}_2"], ["\operatorname{int}_3"], ["\operatorname{out}"]])
        S2 = Matrix([[1], ["x"], ["\operatorname{int}"], ["\operatorname{int}_2"], ["\operatorname{int}_3"], ["\operatorname{out}"]])
        S3 = Matrix([[1], ["x"], ["\operatorname{int}"], ["\operatorname{int}_2"], ["\operatorname{int}_3"], ["\operatorname{out}"]])
        AA = Matrix([["a_{11}"], ["a_{12}"], ["a_{13}"], ["a_{14}"], ["a_{15}"], ["a_{16}"]])
        BB = Matrix([["b_{11}"], ["b_{12}"], ["b_{13}"], ["b_{14}"], ["b_{15}"], ["b_{16}"]])
        CC = Matrix([["c_{11}"], ["c_{12}"], ["c_{13}"], ["c_{14}"], ["c_{15}"], ["c_{16}"]])
        times = Tex("*")
        equals = Tex("=")
        togeth = Group(S, AA, times, S2, BB, equals, S3, CC).arrange()
        parts = togeth.split()
        rectt = Rectangle(width = 2.7, height = 0.6, color = LIGHT).shift(3.9*LEFT + 2*UP)
        rectt2 = Rectangle(width = 2.7, height = 0.6, color = LIGHT).shift(3.9*LEFT + 1.2*UP)
        rect = Rectangle(width = 2.5, height = 0.6, color = LIGHT).shift(3.8*LEFT + 2*UP)
        rect2 = Rectangle(width = 2.5, height = 0.6, color = LIGHT).shift(3.8*LEFT + 1.2*UP)
        rect3 = Rectangle(width = 2.5, height = 0.6, color  = LIGHT).shift(3.8*LEFT + 1.2*DOWN)
        rect4 = Rectangle(width = 2.5, height = 0.6, color  = LIGHT).shift(2*UP)
        rect5 = Rectangle(width = 2.5, height = 0.6, color  = LIGHT).shift(4*RIGHT + 2*DOWN)
        rect6 = Rectangle(width = 2.5, height = 0.6, color  = LIGHT).shift(3.8*LEFT + 0.4*DOWN)
        rect7 = Rectangle(width = 2.5, height = 0.6, color  = LIGHT).shift(2*UP)
        rect8 = Rectangle(width = 2.5, height = 0.6, color  = LIGHT).shift(4*RIGHT + 1.2*DOWN)
        equals_clone = equals.copy().move_to(times.get_center())
        summ = MathTex(r'{{1*a_{11}}}{{+x*a_{12}}}{{\\+\operatorname{int}*a_{13} +\operatorname{int}_2*a_{14}\\+\operatorname{int}_3*a_{15} +\operatorname{out}*a_{16}}}').shift(RIGHT)
        AAA = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[0]).move_to(AA.get_center())
        BBB = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[0]).move_to(BB.get_center())
        CCC = Matrix([["0"], ["0"], ["1"], ["0"], ["0"], ["0"]]).set_color(colours_1[0]).move_to(CC.get_center())
        AAA2 = Matrix([["0"], ["0"], ["1"], ["0"], ["0"], ["0"]]).set_color(colours_1[1]).move_to(AA.get_center())
        BBB2 = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[1]).move_to(BB.get_center())
        CCC2 = Matrix([["0"], ["0"], ["0"], ["1"], ["0"], ["0"]]).set_color(colours_1[1]).move_to(CC.get_center())
        AAA3 = Matrix([["0"], ["1"], ["0"], ["1"], ["0"], ["0"]]).set_color(colours_1[2]).move_to(AA.get_center())
        BBB3 = Matrix([["1"], ["0"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[2]).move_to(BB.get_center())
        CCC3 = Matrix([["0"], ["0"], ["0"], ["0"], ["1"], ["0"]]).set_color(colours_1[2]).move_to(CC.get_center())
        AAA4 = Matrix([["5"], ["0"], ["0"], ["0"], ["1"], ["0"]]).set_color(colours_1[3]).move_to(AA.get_center())
        BBB4 = Matrix([["1"], ["0"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[3]).move_to(BB.get_center())
        CCC4 = Matrix([["0"], ["0"], ["0"], ["0"], ["0"], ["1"]]).set_color(colours_1[3]).move_to(CC.get_center())
        fakeS = Matrix([["1"], ["1"], ["1"], ["1"], ["1"], ["6"]]).set_color(colour_special_darker).move_to(S.get_center())
        fakeS2 = Matrix([["1"], ["1"], ["1"], ["1"], ["1"], ["6"]]).set_color(colour_special_darker).move_to(S2.get_center())
        fakeS3 = Matrix([["1"], ["1"], ["1"], ["1"], ["1"], ["6"]]).set_color(colour_special_darker).move_to(S3.get_center())
        fake_sum = MathTex(r'{{(}}{{5}} {{+}}{{1}}{{)*(}}{{1}}{{)=}}{{6}}').shift(3.5*DOWN)
        fake_sum2 = MathTex(r'{{(}}{{1}} {{+}}{{2}}{{)*(}}{{1}}{{)}}{{=}}{{1}}').shift(2.9*DOWN)
        not_equals = MathTex("\\neq").move_to(fake_sum2[8].get_center()).set_color(RED)
        SS = Matrix([["1"], ["3"], ["9"], ["27"], ["30"], ["35"]]).set_color(colour_special).move_to(S.get_center())
        SS2 = Matrix([["1"], ["3"], ["9"], ["27"], ["30"], ["35"]]).set_color(colour_special).move_to(S2.get_center())
        SS3 = Matrix([["1"], ["3"], ["9"], ["27"], ["30"], ["35"]]).set_color(colour_special).move_to(S3.get_center())
        
        
        self.introduce_constraints(equation, steps, colours_1)
        self.allowed_ops(ops, LIGHT, steps)
        self.introduce_R1CS(vector_compact_1_square, vector_compact, vector_dots, vector_compact_n, A, B, C, vectors)
        self.vector_size(items, LIGHT, vectors, vectorss, steps, vector_compact)
        self.write_dot_prod(items, parts, rectt, rectt2, equals_clone, summ, vectorss, LIGHT)
        self.instantiating_values(AA, BB, CC, AAA, BBB, CCC, AAA2, BBB2, CCC2, AAA3, BBB3, CCC3, AAA4, BBB4, CCC4, steps)
        self.fake_sol(fakeS, fakeS2, fakeS3, S, S2, S3, rect, rect3, rect4, rect5, fake_sum)
        self.fake_sol_2(steps, AAA3, AAA4, BBB3, BBB4, CCC3, CCC4, rect2, rect6, rect7, rect8, fake_sum2, not_equals)
        self.actual_sol(fake_sum, fake_sum2, not_equals, fakeS, fakeS2, fakeS3, SS, SS2, SS3, AA, BB, CC, AAA3, BBB3, CCC3, parts, steps, AAA, BBB, CCC, AAA2, BBB2, CCC2, AAA4, BBB4, CCC4, equation, LIGHT)

    def introduce_constraints(self, equation, steps, colours_1):
        self.play(Write(equation, run_time = 1))
        self.wait(1.5)
        self.play(equation.animate.shift(3.5*UP))
        self.wait(1.5)
        self.play(FadeIn(steps[0]), run_time=1)
        self.wait(1.5)
        self.play(FadeIn(steps[2]), run_time=1)
        self.wait(1.5)
        self.play(FadeIn(steps[4]), run_time=1)
        self.wait(1.5)
        self.play(FadeIn(steps[6]), run_time=1)
        self.wait(1.5)
        self.play(steps.animate.shift(5*LEFT), run_time=1)
        self.wait(1.5)
        self.play(steps[0].animate.set_color(colours_1[0]),
        steps[2].animate.set_color(colours_1[1]),
        steps[4].animate.set_color(colours_1[2]),
        steps[6].animate.set_color(colours_1[3]),
        )
        self.wait(1.5)
        
    def allowed_ops(self, ops, LIGHT, steps):
        # Introduce the allowed operations
        self.play(FadeIn(ops))
        self.wait(1.5)
        self.play(Circumscribe(steps[0], color = LIGHT), Circumscribe(steps[2], color=LIGHT))
        self.wait(1.5)
        self.play(Circumscribe(steps[4], color = LIGHT), Circumscribe(steps[6], color=LIGHT))
        self.wait(1.5)
        self.play(FadeOut(ops))
        self.wait(1.5)

    def introduce_R1CS(self, vector_compact_1_square, vector_compact, vector_dots, vector_compact_n, A, B, C, vectors):
        # Introduce an R1CS

        self.play(FadeIn(vector_compact_1_square))
        self.wait(1.5)
        self.play(*[FadeTransform(vector_compact_1_square[0][s], vectors[s]) for s in range(3)], FadeOut(vector_compact_1_square[1], vector_compact_1_square[2]))
        self.wait(1.5)
        self.play(*[FadeTransform(vectors[s], vector_compact_1_square[0][s]) for s in range(3)], FadeIn(vector_compact_1_square[1], vector_compact_1_square[2]))
        self.wait(1.5)
        self.play(FadeTransform(vector_compact_1_square, vector_compact[0]))
        self.wait(1.5)

        # We are going to have 4 sets of these vectors because we have 4 constraints
        self.play(*[FadeIn(vector_compact[s+1]) for s in range(3)])
        self.wait(1.5)
        
        vector_compact_1_clone = vector_compact[0].copy()
        self.play(FadeTransform(vector_compact_1_clone, vector_compact_1_square))
        self.wait(1.5)
        self.play(*[FadeTransform(vector_compact_1_square[0][s], vectors[s]) for s in range(3)], FadeOut(vector_compact_1_square[1], vector_compact_1_square[2]))
        self.wait(1.5)
        
    def vector_size(self, items, LIGHT, vectors, vectorss, steps, vector_compact):
        # How many items in each vector?
        self.play(FadeIn(items[0], items[2]))
        self.wait(1.5)
        self.play(FadeIn(items[4]))
        self.wait(1.5)
        self.play(FadeIn(items[6]))
        self.wait(1.5)
        self.play(FadeIn(items[8]))
        self.wait(1.5)
        self.play(FadeIn(items[10]))
        self.wait(1.5)
        self.play(FadeIn(items[12], items[14]))
        self.wait(1.5)

        self.play(*[FadeTransform(vectors[s], vectorss[s]) for s in range(3)])
        self.wait(1.5)
        self.play(Circumscribe(steps[0], color = LIGHT))
        self.wait(1.5)
        self.play(FadeOut(steps[2], steps[4], steps[6]), *[FadeOut(vector_compact[s]) for s in range(4)], steps[0].animate.shift(2.3*UP))
        self.wait(1.5)

    def write_dot_prod(self, items, parts, rectt, rectt2, equals_clone, summ, vectorss, LIGHT):
        self.play(FadeIn(parts[0]), FadeOut(items), FadeTransform(vectorss[0], parts[1]), FadeOut(vectorss[1], vectorss[2]))
        self.wait(1.5)

        # rectangles used to highlight how we do the dot product in the first vector pair
        other_rects = [2,3,4,5]
        for s in range(4):
            other_rects[s] = Rectangle(width = 2.7, height = 0.6, color = LIGHT).shift(3.9*LEFT + 0.4*UP + s*0.8*DOWN)

        # aligning vectors
        self.play(Write(equals_clone))
        self.wait(1.5)
        self.play(Write(rectt))
        self.wait(1.5)
        self.play(FadeIn(summ[0]))
        self.wait(1.5)
        self.play(Write(rectt2))
        self.wait(1.5)
        self.play(FadeIn(summ[2]))
        self.wait(1.5)
        self.play(FadeIn(summ[4]), *[FadeIn(other_rects[s]) for s in range(4)])
        self.wait(1.5)
        self.play(FadeOut(summ, equals_clone, rectt, rectt2), *[FadeOut(other_rects[s]) for s in range(4)])
        self.wait(1.5)
        self.play(FadeIn(parts[2], parts[3], parts[4]))
        self.wait(1.5)
        self.play(FadeIn(parts[5], parts[6], parts[7]))
        self.wait(1.5)

    def instantiating_values(self, AA, BB, CC, AAA, BBB, CCC, AAA2, BBB2, CCC2, AAA3, BBB3, CCC3, AAA4, BBB4, CCC4, steps):
        # Instantiating actual values for the sets of vectors
        self.play(FadeTransform(AA, AAA), FadeTransform(BB, BBB), FadeTransform(CC, CCC))
        self.wait(1.5)
        steps[2].move_to(steps[0].get_center())
        self.play(FadeTransform(steps[0], steps[2]), FadeTransform(AAA, AAA2), FadeTransform(BBB, BBB2), FadeTransform(CCC, CCC2))
        self.wait(1.5)
        steps[4].move_to(steps[0].get_center())
        self.play(FadeTransform(steps[2], steps[4]), FadeTransform(AAA2, AAA3), FadeTransform(BBB2, BBB3), FadeTransform(CCC2, CCC3))
        self.wait(1.5)
        steps[6].move_to(steps[0].get_center())
        self.play(FadeTransform(steps[4], steps[6]), FadeTransform(AAA3, AAA4), FadeTransform(BBB3, BBB4), FadeTransform(CCC3, CCC4))
        self.wait(1.5)

    def fake_sol(self, fakeS, fakeS2, fakeS3, S, S2, S3, rect, rect3, rect4, rect5, fake_sum):
        # Attempted solution vector
        self.play(FadeTransform(S, fakeS), FadeTransform(S2, fakeS2), FadeTransform(S3, fakeS3))
        self.wait(1.5)

        # numberplane = NumberPlane()
        # self.add(numberplane)
        self.wait(1.5)
        self.play(Write(rect), Write(rect3), Write(rect4), Write(rect5))
        self.wait(1.5)
        unfaded = [0,2,3,5,7]
        self.play(FadeTransform(rect, fake_sum[1]), FadeTransform(rect3, fake_sum[4]), FadeTransform(rect4, fake_sum[6]), FadeTransform(rect5, fake_sum[8]), *[FadeIn(fake_sum[s]) for s in unfaded])
        self.wait(1.5)

    def fake_sol_2(self, steps, AAA3, AAA4, BBB3, BBB4, CCC3, CCC4, rect2, rect6, rect7, rect8, fake_sum2, not_equals):
        self.play(FadeTransform(AAA4, AAA3), FadeTransform(BBB4, BBB3), FadeTransform(CCC4, CCC3), FadeTransform(steps[6], steps[4]))
        self.wait(1.5)
        self.play(Write(rect2), Write(rect6), Write(rect7), Write(rect8))
        self.wait(1.5)
        unfaded = [0,2,3,5,7,8]
        self.play(FadeTransform(rect2, fake_sum2[1]), FadeTransform(rect6, fake_sum2[4]), FadeTransform(rect7, fake_sum2[6]), FadeTransform(rect8, fake_sum2[9]), *[FadeIn(fake_sum2[s]) for s in unfaded])
        self.wait(1.5)
        self.play(FadeTransform(fake_sum2[8], not_equals))
        self.wait(1.5)

    def actual_sol(self, fake_sum, fake_sum2, not_equals, fakeS, fakeS2, fakeS3, SS, SS2, SS3, AA, BB, CC, AAA3, BBB3, CCC3, parts, steps, AAA, BBB, CCC, AAA2, BBB2, CCC2, AAA4, BBB4, CCC4, equation, LIGHT):
        self.play(FadeTransform(fakeS, SS), FadeTransform(fakeS2, SS2), FadeTransform(fakeS3, SS3), FadeTransform(AAA3, AA), FadeTransform(BBB3, BB), FadeTransform(CCC3, CC), FadeOut(fake_sum, fake_sum2, not_equals, steps[4]))
        self.wait(1.5)
        self.play(SS.animate.shift(LEFT), FadeOut(SS2, SS3, parts[2], parts[5], AA, BB, CC))
        self.wait(1.5)
        constraints = Group(steps[0], steps[2], steps[4], steps[6]).arrange_in_grid(buff=1.5).move_to(2.7*RIGHT)
        self.play(FadeIn(constraints))
        self.wait(1.5)

        constraint1 = Group(AAA, BBB, CCC).arrange().scale(0.7)
        constraint2 = Group(AAA2, BBB2, CCC2).arrange().scale(0.7)
        constraint3 = Group(AAA3, BBB3, CCC3).arrange().scale(0.7)
        constraint4 = Group(AAA4, BBB4, CCC4).arrange().scale(0.7)
        constraints_real = Group(constraint1, constraint2, constraint3, constraint4).arrange_in_grid().move_to(3*RIGHT)
        self.play(*[FadeTransform(constraints[s], constraints_real[s]) for s in range(4)], equation.animate.shift(2*LEFT))
        self.wait(1.5)

        # Public knowledge
        pub_param_1 = SurroundingRectangle(constraints_real, color = LIGHT)
        pub_param_2 = SurroundingRectangle(SS[0][5], color = LIGHT)
        initial_equ = SurroundingRectangle(equation, color = LIGHT)
        zero_knowledge = SurroundingRectangle(SS[0][0:5], color = PURPLE)
        self.play(Write(pub_param_1))
        self.wait(1.5)
        self.play(Write(pub_param_2))
        self.wait(1.5)
        self.play(Write(initial_equ))
        self.wait(1.5)
        self.play(Write(zero_knowledge))
        self.wait(1.5)
        self.play(zero_knowledge.animate.set_fill(PURPLE, opacity=1.0))
        self.wait(1.5)
        generic_equ = MathTex("x^{3} + x + 5 = y", color=YELLOW).move_to(equation)
        self.play(pub_param_2.animate.set_fill(LIGHT, opacity=1.0), FadeTransform(equation, generic_equ))
        self.wait(1.5)
        self.play(pub_param_2.animate.set_fill(LIGHT, opacity=0), FadeTransform(generic_equ, equation))
        self.wait(1.5)

        # Fade out everything and remind ourselves how we got here
        self.play(FadeOut(constraints_real, pub_param_1, pub_param_2, initial_equ, zero_knowledge, SS, equation))
        self.wait(1.5)
        self.play(Write(equation))
        self.wait(1.5)
        self.play(*[Write(constraints_real[s][0]) for s in range(4)], *[Write(constraints_real[s][1]) for s in range(4)], *[Write(constraints_real[s][2]) for s in range(4)])
        self.wait(1.5)
        self.play(Write(SS))
        self.wait(1.5)



def langrage_basis(x, s):
    result = 1
    for t in range(4):
            if t != s:
                result *= (x - (t+1))/((s+1) - (t+1))
    return result

class Lagrange(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        LIGHT = "#00c0f9"
        DARK = "#154bf9"
        colour_special = "#931CFF"
        colour_special_darker = "#9C7900"
        colours_1 = ["#FFCC17", "#FF5555", "#E561E5", "#FF7D54"]
        colours_2 = ["#A479E0", "#4DECAA", "#FFD954", "#22F1E4"]
        colour_lagr = "#0F915B"

        lagrange_axes = Axes(x_range=[0,8],y_range=[0,6], y_length=5, tips=False, axis_config={"include_numbers": True}).shift(0.8*UP)
        vec = [[1, 3, 5, 1]]
        vecc = Matrix(vec).shift(3.5*DOWN + 2.3*LEFT)
        points = [Dot(lagrange_axes.coords_to_point(s+1,vec[0][s])) for s in range(4)]
        lagrange_bases1 = lambda x: langrage_basis(x, 0)
        lagrange_bases2 = lambda x: langrage_basis(x, 1)
        lagrange_bases3 = lambda x: langrage_basis(x, 2)
        lagrange_bases4 = lambda x: langrage_basis(x, 3)
        lagrangey = [lagrange_bases1, lagrange_bases2, lagrange_bases3, lagrange_bases4]
        plots = [*[lagrange_axes.plot(lagrangey[i], color = colours_2[i]) for i in range(4)]]
        lagr_equations = [0,1,2,3]
        lagr_succinct = [0,1,2,3]
        lagr_equations[0] = MathTex("-\\frac{1}{6}x^3+\\frac{3}{2}x^2-\\frac{13}{3}x+4").set_color(colours_2[0])
        lagr_equations[1] = MathTex("\\frac{1}{2}x^3 - 4x^2 + \\frac{19}{2}x - 6").set_color(colours_2[1])
        lagr_equations[2] = MathTex("-\\frac{1}{2}x^3 + \\frac{7}{2}x^2 - 7x + 4").set_color(colours_2[2])
        lagr_equations[3] = MathTex("-\\frac{1}{6}x^3+\\frac{3}{2}x^2-\\frac{13}{3}x+4").set_color(colours_2[3])
        lagr_succinct[0] = MathTex("L_1(x)").set_color(colours_2[0]).move_to(4*RIGHT)
        lagr_succinct[1] = MathTex("L_2(x)").set_color(colours_2[1]).move_to(4*RIGHT)
        lagr_succinct[2] = MathTex("L_3(x)").set_color(colours_2[2]).move_to(4*RIGHT)
        lagr_succinct[3] = MathTex("L_4(x)").set_color(colours_2[3]).move_to(4*RIGHT)
        lagr_equations_with_succinct = [0,1,2,3]
        for s in range(4):
            lagr_equations_with_succinct[s] = Group(lagr_equations[s], lagr_succinct[s]).arrange(DOWN).shift(4*RIGHT)
        multiply_to_fit = [0,1]
        multiply_to_fit[0] = MathTex("3*(\\frac{1}{2}x^3 - 4x^2 + \\frac{19}{2}x - 6)").set_color(colours_2[1])
        multiply_to_fit[1] = MathTex("5*(-\\frac{1}{2}x^3 + \\frac{7}{2}x^2 - 7x + 4)").set_color(colours_2[2])
        multiply_to_fit_succinct = [0,1]
        multiply_to_fit_succinct[0] = Group(multiply_to_fit[0], MathTex("3*L_2(x)").set_color(colours_2[1])).arrange(DOWN).shift(4*RIGHT)
        multiply_to_fit_succinct[1] = Group(multiply_to_fit[1], MathTex("5*L_3(x)").set_color(colours_2[2])).arrange(DOWN).shift(4*RIGHT)
        sum_of_lagr_for_interp_example = Group(lagr_succinct[0].copy(), MathTex("+3*L_2(x)").set_color(colours_2[1]), MathTex("+5*L_3(x)").set_color(colours_2[2]), MathTex("+L_4(x)").set_color(colours_2[3])).arrange(DOWN).shift(4*RIGHT)

        interpoly = lambda x: vec[0][0]*langrage_basis(x, 0)+vec[0][1]*langrage_basis(x, 1)+vec[0][2]*langrage_basis(x, 2)+vec[0][3]*langrage_basis(x, 3)
        interpoly_plot = lagrange_axes.plot(interpoly, color = BLUE)
        interpoly_equ = MathTex("-x^3 + 6x^2 - 9x + 5").set_color(BLUE).move_to(sum_of_lagr_for_interp_example.get_center())
        interpo_point = Dot(lagrange_axes.coords_to_point(0,5), color = RED)
        vecc_basis_2 = lambda x: vec[0][1]*langrage_basis(x, 1)
        vecc_basis_3 = lambda x: vec[0][2]*langrage_basis(x, 2)
        vecc_plots = [lagrange_axes.plot(vecc_basis_2, color = colours_2[1]), lagrange_axes.plot(vecc_basis_3, color = colours_2[2])]

        AAA = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[0])
        BBB = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[0])
        CCC = Matrix([["0"], ["0"], ["1"], ["0"], ["0"], ["0"]]).set_color(colours_1[0])
        AAA2 = Matrix([["0"], ["0"], ["1"], ["0"], ["0"], ["0"]]).set_color(colours_1[1])
        BBB2 = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[1])
        CCC2 = Matrix([["0"], ["0"], ["0"], ["1"], ["0"], ["0"]]).set_color(colours_1[1])
        AAA3 = Matrix([["0"], ["1"], ["0"], ["1"], ["0"], ["0"]]).set_color(colours_1[2])
        BBB3 = Matrix([["1"], ["0"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[2])
        CCC3 = Matrix([["0"], ["0"], ["0"], ["0"], ["1"], ["0"]]).set_color(colours_1[2])
        AAA4 = Matrix([["5"], ["0"], ["0"], ["0"], ["1"], ["0"]]).set_color(colours_1[3])
        BBB4 = Matrix([["1"], ["0"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[3])
        CCC4 = Matrix([["0"], ["0"], ["0"], ["0"], ["0"], ["1"]]).set_color(colours_1[3])

        vec2 = [[0, 0, 0, 5]]
        vecc2 = Matrix(vec2).set_column_colors(colours_1[0], colours_1[1], colours_1[2], colours_1[3]).shift(3.5*DOWN + 2.3*LEFT)
        firs_a_vec_function = lambda x: vec2[0][3]*langrage_basis(x, 3)
        first_a_vec = lagrange_axes.plot(firs_a_vec_function, color = GREEN)
        a_vec = Group(AAA.copy(), AAA2.copy(), AAA3.copy(), AAA4.copy()).arrange().move_to(3.5*RIGHT)
        A1_lagr = MathTex("\\frac{5}{6}x^3 - 5x^2 + \\frac{55}{6}x - 5").shift(3.5*RIGHT).set_color(GREEN)

        constraint1 = Group(AAA, BBB, CCC).arrange().scale(0.7)
        constraint2 = Group(AAA2, BBB2, CCC2).arrange().scale(0.7)
        constraint3 = Group(AAA3, BBB3, CCC3).arrange().scale(0.7)
        constraint4 = Group(AAA4, BBB4, CCC4).arrange().scale(0.7)
        constraints_real = Group(constraint1, constraint2, constraint3, constraint4).arrange_in_grid().move_to(3*RIGHT)
        copies = Group(AAA.copy(), AAA2.copy(), AAA3.copy(), AAA4.copy())

        steps = [0,1,2,3]
        steps[0] = MathTex("\operatorname{int} = x*x").set_color(colours_1[0]).move_to(constraints_real[0].get_center())
        steps[1] = MathTex("\operatorname{int}_2 = \operatorname{int}*x").set_color(colours_1[1]).move_to(constraints_real[1].get_center())
        steps[2] = MathTex("\operatorname{int}_3 = \operatorname{int}_2 + x").set_color(colours_1[2]).move_to(constraints_real[2].get_center())
        steps[3] = MathTex("\operatorname{out} = \operatorname{int}_3 + 5").set_color(colours_1[3]).move_to(constraints_real[3].get_center())
        steps_group = Group(steps[0], steps[1], steps[2], steps[3]).arrange(DOWN).move_to(3*RIGHT)
        equation = MathTex("x^{3} + x + 5 = 35", color=colour_special)

        A2_lagr = MathTex("-\\frac{2}{3}x^3 + 5x^2 + -\\frac{34}{3}x + 8").scale(0.6)
        A3_lagr = MathTex("\\frac{1}{2}x^3 - 4x^2 + \\frac{19}{2}x - 6").scale(0.6)
        A4_lagr = MathTex("-x^3 + 7x^2 + -14x + 8").scale(0.6)
        A5_lagr = MathTex("\\frac{1}{6}x^3 - x^2 + \\frac{11}{6}x - 1").scale(0.6)
        A6_lagr = MathTex("0x^3 + 0x^2 + 0x + 0").scale(0.6)
        A = [A1_lagr, A2_lagr, A3_lagr, A4_lagr, A5_lagr, A6_lagr]
        for s in range(5):
            A[s+1].move_to(2.5*LEFT + 2*UP + 0.8*(s+1)*DOWN).set_color(GREEN)
        A_interp_succinct = [0,1,2,3,4,5]
        A_interp_succinct[0] = MathTex("A_{1}(x)").scale(0.7).move_to(A[1].get_center() + 0.8*UP).set_color(BLUE)
        A_interp_succinct[1] = MathTex("A_{2}(x)").scale(0.7).move_to(A[1].get_center()).set_color(BLUE)
        A_interp_succinct[2] = MathTex("A_{3}(x)").scale(0.7).move_to(A[2].get_center()).set_color(BLUE)
        A_interp_succinct[3] = MathTex("A_{4}(x)").scale(0.7).move_to(A[3].get_center()).set_color(BLUE)
        A_interp_succinct[4] = MathTex("A_{5}(x)").scale(0.7).move_to(A[4].get_center()).set_color(BLUE)
        A_interp_succinct[5] = MathTex("A_{6}(x)").scale(0.7).move_to(A[5].get_center()).set_color(BLUE)
        B_interp_succinct = [0,1,2,3,4,5]
        B_interp_succinct[0] = MathTex("B_{1}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct[1] = MathTex("B_{2}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct[2] = MathTex("B_{3}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct[3] = MathTex("B_{4}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct[4] = MathTex("B_{5}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct[5] = MathTex("B_{6}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct = [0,1,2,3,4,5]
        C_interp_succinct[0] = MathTex("C_{1}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct[1] = MathTex("C_{2}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct[2] = MathTex("C_{3}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct[3] = MathTex("C_{4}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct[4] = MathTex("C_{5}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct[5] = MathTex("C_{6}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct_group = Group(B_interp_succinct[0], B_interp_succinct[1], B_interp_succinct[2], B_interp_succinct[3], B_interp_succinct[4], B_interp_succinct[5]).scale(0.7).arrange(DOWN, buff=0.33).move_to(2*RIGHT)
        C_interp_succinct_group = Group(C_interp_succinct[0], C_interp_succinct[1], C_interp_succinct[2], C_interp_succinct[3], C_interp_succinct[4], C_interp_succinct[5]).scale(0.7).arrange(DOWN, buff=0.32).move_to(6.3*RIGHT)

        L_polys = [0,1,2,3]
        L_polys[0] = MathTex("L_{1}(x)").scale(0.7).move_to(a_vec[0].get_top() + 0.6*UP).set_color(colour_lagr)
        L_polys[1] = MathTex("L_{2}(x)").scale(0.7).move_to(a_vec[1].get_top() + 0.6*UP).set_color(colour_lagr)
        L_polys[2] = MathTex("L_{3}(x)").scale(0.7).move_to(a_vec[2].get_top() + 0.6*UP).set_color(colour_lagr)
        L_polys[3] = MathTex("L_{4}(x)").scale(0.7).move_to(a_vec[3].get_top() + 0.6*UP).set_color(colour_lagr)

        self.unique_polys(lagrange_axes, vecc, points, plots, interpoly_plot, interpoly_equ, interpo_point, vecc_plots, lagr_equations_with_succinct, multiply_to_fit_succinct, sum_of_lagr_for_interp_example)
        self.making_polys(lagrange_axes, vec2, vecc2, first_a_vec, plots, a_vec, A1_lagr, constraints_real, copies, colours_1, colours_2, lagr_succinct)
        self.return_to_vecs(a_vec, vecc2, A1_lagr, LIGHT, A_interp_succinct, A, L_polys, colour_lagr)
        self.remind_where_vec_came_from(a_vec, constraints_real, steps_group, equation, A_interp_succinct, L_polys, LIGHT, B_interp_succinct_group, C_interp_succinct_group, colour_special, colours_1)

    def unique_polys(self, lagrange_axes, vecc, points, plots, interpoly_plot, interpoly_equ, interpo_point, vecc_plots, lagr_equations_with_succinct, multiply_to_fit_succinct, sum_of_lagr_for_interp_example):
        self.add(lagrange_axes)
        self.wait(1.5)
        self.play(FadeIn(vecc))
        self.wait(1.5)
        vecc_copy = vecc.copy()
        self.play(*[FadeTransform(vecc_copy[0][s], points[s]) for s in range(4)])
        self.wait(1.5)

        # show basis polys
        self.play(FadeIn(plots[0], lagr_equations_with_succinct[0]))
        self.wait(1.5)
        self.play(FadeTransform(plots[0], plots[1]), *[FadeTransform(lagr_equations_with_succinct[0][s], lagr_equations_with_succinct[1][s]) for s in range(2)])
        self.wait(1.5)
        self.play(FadeTransform(plots[1], vecc_plots[0]), *[FadeTransform(lagr_equations_with_succinct[1][s], multiply_to_fit_succinct[0][s]) for s in range(2)])
        self.wait(1.5)
        self.play(FadeTransform(vecc_plots[0], plots[2]), *[FadeTransform(multiply_to_fit_succinct[0][s], lagr_equations_with_succinct[2][s]) for s in range(2)])
        self.wait(1.5)
        self.play(FadeTransform(plots[2], vecc_plots[1]), *[FadeTransform(lagr_equations_with_succinct[2][s], multiply_to_fit_succinct[1][s]) for s in range(2)])
        self.wait(1.5)
        self.play(FadeTransform(vecc_plots[1], plots[3]), *[FadeTransform(multiply_to_fit_succinct[1][s], lagr_equations_with_succinct[3][s]) for s in range(2)])
        self.wait(1.5)

        self.play(FadeIn(plots[0], vecc_plots[0], vecc_plots[1], sum_of_lagr_for_interp_example), FadeOut(lagr_equations_with_succinct[3]))
        self.wait(1.5)
        self.play(FadeOut(plots[0], plots[3], vecc_plots[0], vecc_plots[1]))
        self.play(Create(interpoly_plot))
        self.wait(1.5)
        self.play(*[Unwrite(sum_of_lagr_for_interp_example[s]) for s in range(4)])
        self.play(Write(interpoly_equ))
        self.wait(1.5)
        self.play(FadeOut(lagrange_axes, interpoly_equ, interpoly_plot, vecc), *[FadeOut(points[s]) for s in range(4)])
        self.wait(1.5)

    def making_polys(self, lagrange_axes, vec2, vecc2, first_a_vec, plots, a_vec, A1_lagr, constraints_real, copies, colours_1, colours_2, lagr_succinct):
        self.play(FadeIn(constraints_real))
        self.wait(1.5)
        self.add(copies)
        self.play(FadeOut(constraints_real))
        self.wait(1.5)

        # Make the grid of A vecs
        self.play(*[FadeTransform(copies[s], a_vec[s]) for s in range(4)])
        self.wait(1.5)
        self.play(*[FadeTransform(a_vec[s][0][0].copy(), vecc2[0][s]) for s in range(4)], FadeIn(vecc2[1], vecc2[2]), FadeOut(a_vec))
        self.wait(1.5)
        self.play(FadeIn(lagrange_axes))
        self.wait(1.5)
        vecc2_copy = vecc2.copy()
        new_points = [Dot(lagrange_axes.coords_to_point(s+1,vec2[0][s]), color=colours_1[s]) for s in range(4)]
        self.play(*[FadeTransform(vecc2_copy[0][s], new_points[s]) for s in range(4)])
        self.wait(1.5)

        # show bases again alongside L_i(x) and what multiple is needed of each
        lagr_4_times_five = MathTex("5*L_4(x)").set_color(GREEN).shift(4*RIGHT)
        self.play(FadeIn(plots[0], lagr_succinct[0].move_to(lagr_4_times_five.get_center())))
        self.wait(1.5)
        self.play(FadeTransform(plots[0], plots[1]), FadeTransform(lagr_succinct[0], lagr_succinct[1].move_to(lagr_4_times_five.get_center())))
        self.wait(1.5)
        self.play(FadeTransform(plots[1], plots[2]), FadeTransform(lagr_succinct[1], lagr_succinct[2].move_to(lagr_4_times_five.get_center())))
        self.wait(1.5)
        self.play(FadeTransform(plots[2], plots[3]), FadeTransform(lagr_succinct[2], lagr_succinct[3].move_to(lagr_4_times_five.get_center())))
        self.wait(1.5)
        self.play(FadeTransform(plots[3], first_a_vec), FadeTransform(lagr_succinct[3], lagr_4_times_five))
        self.wait(1.5)
        self.play(FadeTransform(lagr_4_times_five, A1_lagr))
        self.wait(1.5)
        self.play(FadeOut(lagrange_axes, *[new_points[s] for s in range(4)], first_a_vec))
        self.wait(1.5)
        self.play(A1_lagr.animate.shift(6*LEFT+ 2*UP).scale(0.6))
        self.wait(1.5)

    def return_to_vecs(self, a_vec, vecc2, A1_lagr, LIGHT, A_interp_succinct, A, L_polys, colour_lagr):
        a_vec_top_copies = [0,1,2,3]
        for s in range(4):
            a_vec_top_copies[s] = a_vec[s][0][0].copy()
        self.play(*[FadeTransform(vecc2[0][s], a_vec_top_copies[s]) for s in range(4)], FadeIn(a_vec), FadeOut(vecc2[1], vecc2[2]))
        self.wait(1.5)
        self.play(*[FadeOut(a_vec_top_copies[s]) for s in range(4)])


        Ai_values_rectangles = [0,1,2,3,4,5] 
        for s in range(6):
            Ai_values_rectangles[s] = Rectangle(width = 4.5, height = 0.5, color =LIGHT).move_to(3.5*RIGHT + 2*UP + 0.8*(s)*DOWN)
        self.play(Create(Ai_values_rectangles[0]))
        self.wait(1.5)
        self.play(ReplacementTransform(Ai_values_rectangles[0], A1_lagr))
        self.wait(1.5)
        self.play(*[FadeIn(Ai_values_rectangles[s+1]) for s in range(5)])
        self.wait(1.5)

        self.play(*[Write(L_polys[s]) for s in range(4)])
        self.wait(1.5)

        # present the interpolations in Lagrange basis
        interpolation_in_lagr_basis = [0,1,2,3,4]
        interpolation_in_lagr_basis[0] = MathTex("1*L_{1}(x)+1*L_{3}(x)").scale(0.7).move_to(A[1].get_center()).set_color(colour_lagr)
        interpolation_in_lagr_basis[1] = MathTex("1*L_{2}(x)").scale(0.7).move_to(A[2].get_center()).set_color(colour_lagr)
        interpolation_in_lagr_basis[2] = MathTex("2*L_{3}(x)").scale(0.7).move_to(A[3].get_center()).set_color(colour_lagr)
        interpolation_in_lagr_basis[3] = MathTex("1*L_{4}(x)").scale(0.7).move_to(A[4].get_center()).set_color(colour_lagr)
        interpolation_in_lagr_basis[4] = MathTex("0").scale(0.7).move_to(A[5].get_center()).set_color(colour_lagr)

        self.play(*[FadeTransform(Ai_values_rectangles[s+1], interpolation_in_lagr_basis[s]) for s in range(5)])
        self.wait(1.5)
        
        # introduce the interpolated A equations in x basis
        self.play(*[FadeTransform(interpolation_in_lagr_basis[s], A[s+1]) for s in range(5)])
        self.wait(1.5)


        self.play(*[FadeTransform(A[s], A_interp_succinct[s]) for s in range(6)])
        self.wait(1.5)

        # plane = NumberPlane()
        # self.add(plane)
        self.wait(1.5)
        self.play(*[FadeOut(L_polys[s]) for s in range(4)])
        self.wait(1.5)


    def remind_where_vec_came_from(self, a_vec, constraints_real, steps_group, equation, A_interp_succinct, L_polys, LIGHT, B_interp_succinct_group, C_interp_succinct_group, colour_special, colours_1):
        constraints_real_first_copy = [0,1,2,3]
        for s in range(4):
            constraints_real_first_copy[s] = constraints_real[s][0].copy()
        self.play(*[FadeTransform(a_vec[s], constraints_real_first_copy[s]) for s in range(4)], FadeIn(constraints_real), *[FadeOut(A_interp_succinct[s]) for s in range(6)])
        self.play(*[FadeOut(constraints_real_first_copy[s]) for s in range(4)])
        self.wait(1.5)

        # link vector triples to constraints
        surrounder = [0,1,2,3]
        surrounder[0] = SurroundingRectangle(constraints_real[0], color = colours_1[0])
        surrounder[1] = SurroundingRectangle(constraints_real[1], color = colours_1[1])
        surrounder[2] = SurroundingRectangle(constraints_real[2], color = colours_1[2])
        surrounder[3] = SurroundingRectangle(constraints_real[3], color = colours_1[3])
        self.play(*[FadeTransform(surrounder[s], steps_group[s]) for s in range(4)], *[FadeOut(constraints_real[s]) for s in range(4)])
        self.wait(1.5)

        # link to original equation
        final_surrounder = SurroundingRectangle(steps_group, color = colour_special)
        self.play(FadeTransform(final_surrounder, equation.move_to(steps_group.get_center() + 3*UP)))
        self.wait(1.5)

        # create constraints from equations, and turn those constraints into vector triples
        self.play(FadeTransform(equation, final_surrounder))
        self.play(FadeOut(final_surrounder))
        self.wait(1.5)
        self.play(*[FadeTransform(steps_group[s], surrounder[s]) for s in range(4)], *[FadeIn(constraints_real[s]) for s in range(4)])
        self.play(*[FadeOut(surrounder[s]) for s in range(4)])
        self.wait(1.5)
        sol_vec = Matrix([[1], ["x"], ["\operatorname{int}"], ["\operatorname{int}_2"], ["\operatorname{int}_3"], ["\operatorname{out}"]]).set_color(colour_special).move_to(5*LEFT)
        self.play(FadeIn(sol_vec))
        self.wait(1.5)
        self.play(FadeOut(sol_vec))
        self.wait(1.5)

        a_vecs = Group(constraints_real[0][0].copy(), constraints_real[1][0].copy(), constraints_real[2][0].copy(), constraints_real[3][0].copy()).arrange(buff = 0.1)
        b_vecs = Group(constraints_real[0][1].copy(), constraints_real[1][1].copy(), constraints_real[2][1].copy(), constraints_real[3][1].copy()).arrange(buff = 0.1)
        c_vecs = Group(constraints_real[0][2].copy(), constraints_real[1][2].copy(), constraints_real[2][2].copy(), constraints_real[3][2].copy()).arrange(buff = 0.1)
        vec_groups = Group(a_vecs, b_vecs, c_vecs).arrange(buff = 1.3)
        self.play(*[FadeTransform(constraints_real[s][0], a_vecs[s]) for s in range(4)], *[FadeTransform(constraints_real[s][1], b_vecs[s]) for s in range(4)], *[FadeTransform(constraints_real[s][2], c_vecs[s]) for s in range(4)])
        self.wait(1.5)

        L_basis_group = Group(L_polys[0], L_polys[1], L_polys[2], L_polys[3]).scale(0.7).arrange(buff = 0.18).move_to(a_vecs.get_center() + 2*UP)
        self.play(*[Write(L_basis_group[s]) for s in range(4)])
        self.wait(1.5)

        reccys = [0,1,2,3,4,5]
        for s in range(6):
            reccys[s] = Rectangle(width = 2.8, height = 0.4, color = LIGHT).move_to(a_vecs.get_center() + 1.4*UP + s*0.56*DOWN)
        reccys2 = [0,1,2,3,4,5]
        for s in range(6):
            reccys2[s] = Rectangle(width = 2.8, height = 0.4, color = LIGHT).move_to(b_vecs.get_center() + 1.4*UP + s*0.56*DOWN)
        reccys3 = [0,1,2,3,4,5]
        for s in range(6):
            reccys3[s] = Rectangle(width = 2.8, height = 0.4, color = LIGHT).move_to(c_vecs.get_center() + 1.4*UP + s*0.56*DOWN)

        self.play(*[FadeIn(reccys[s]) for s in range(6)])
        self.wait(1.5)

        
        A_interp_succinct_group = Group(A_interp_succinct[0], A_interp_succinct[1], A_interp_succinct[2], A_interp_succinct[3], A_interp_succinct[4], A_interp_succinct[5]).scale(0.7).arrange(DOWN, buff=0.33).move_to(2.4*LEFT)
        self.play(*[FadeTransform(reccys[s], A_interp_succinct_group[s]) for s in range(6)])
        self.wait(1.5)

        # Display Lagrange bases above the b and c vec groups
        L_basis_group2 = L_basis_group.copy().move_to(b_vecs.get_center() + 2*UP)
        L_basis_group3 = L_basis_group.copy().move_to(c_vecs.get_center() + 2*UP)
        self.play(*[Write(L_basis_group2[s]) for s in range(4)], *[Write(L_basis_group3[s]) for s in range(4)])
        self.wait(1.5)

        # Rectangles around b_vec and c_vec entries
        self.play(*[FadeIn(reccys2[s]) for s in range(6)], *[FadeIn(reccys3[s]) for s in range(6)])
        self.wait(1.5)
        self.play(*[ReplacementTransform(reccys2[s], B_interp_succinct_group[s]) for s in range(6)], *[ReplacementTransform(reccys3[s], C_interp_succinct_group[s]) for s in range(6)])
        self.wait(1.5)

        # Fade out constraint vectors and Lagrange bases
        self.play(FadeOut(L_basis_group, L_basis_group2, L_basis_group3, a_vecs, b_vecs, c_vecs))
        self.wait(1.5)

        # Make lagr interp into actual vectors rather than just grouping, and bring in sol vec again
        A_lagr_vec = Matrix([["A_1(x)"], ["A_2(x)"], ["A_3(x)"], ["A_4(x)"], ["A_5(x)"], ["A_6(x)"]])
        B_lagr_vec = Matrix([["B_1(x)"], ["B_2(x)"], ["B_3(x)"], ["B_4(x)"], ["B_5(x)"], ["B_6(x)"]])
        C_lagr_vec = Matrix([["C_1(x)"], ["C_2(x)"], ["C_3(x)"], ["C_4(x)"], ["C_5(x)"], ["C_6(x)"]])
        times = Tex("*")
        equals = Tex("=")
        minus = Tex("-")
        zero = Tex("0")
        combined_vectors = Group(sol_vec, A_lagr_vec, times, sol_vec.copy(), B_lagr_vec, equals, sol_vec.copy(), C_lagr_vec).arrange()
        combined_vectors_rearranged = Group(sol_vec.copy(), A_lagr_vec.copy(), times.copy(), sol_vec.copy(), B_lagr_vec.copy(), minus.copy(), sol_vec.copy(), C_lagr_vec.copy(), equals.copy(), zero).arrange()
        self.play(*[FadeTransform(A_interp_succinct_group[s], A_lagr_vec[0][s]) for s in range(6)], *[FadeTransform(B_interp_succinct_group[s], B_lagr_vec[0][s]) for s in range(6)], *[FadeTransform(C_interp_succinct_group[s], C_lagr_vec[0][s]) for s in range(6)], FadeIn(A_lagr_vec[1], A_lagr_vec[2], B_lagr_vec[1], B_lagr_vec[2], C_lagr_vec[1], C_lagr_vec[2]))
        self.wait(1.5)
        self.play(FadeIn(combined_vectors[0], combined_vectors[2], combined_vectors[3], combined_vectors[5], combined_vectors[6]))
        self.wait(1.5)
        self.play(*[FadeTransform(combined_vectors[s], combined_vectors_rearranged[s]) for s in range(8)], FadeIn(combined_vectors_rearranged[8], combined_vectors_rearranged[9]))
        self.wait(1.5)

        # evaluating the poly at 1 returns us to the first constraints
        A_lagr_vec_at_zero = Matrix([["A_1(1)"], ["A_2(1)"], ["A_3(1)"], ["A_4(1)"], ["A_5(1)"], ["A_6(1)"]]).set_color(colours_1[0]).move_to(A_lagr_vec.get_center())
        B_lagr_vec_at_zero = Matrix([["B_1(1)"], ["B_2(1)"], ["B_3(1)"], ["B_4(1)"], ["B_5(1)"], ["B_6(1)"]]).set_color(colours_1[0]).move_to(B_lagr_vec.get_center())
        C_lagr_vec_at_zero = Matrix([["C_1(1)"], ["C_2(1)"], ["C_3(1)"], ["C_4(1)"], ["C_5(1)"], ["C_6(1)"]]).set_color(colours_1[0]).move_to(C_lagr_vec.get_center())
        grouped_at_zero = Group(sol_vec.copy(), A_lagr_vec_at_zero.copy(), times.copy(), sol_vec.copy(), B_lagr_vec_at_zero.copy(), minus.copy(), sol_vec.copy(), C_lagr_vec_at_zero.copy(), equals.copy(), zero.copy()).arrange()
        A_lagr_vec_at_zero_value = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[0]).move_to(A_lagr_vec.get_center())
        B_lagr_vec_at_zero_value = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[0]).move_to(B_lagr_vec.get_center())
        C_lagr_vec_at_zero_value = Matrix([["0"], ["0"], ["1"], ["0"], ["0"], ["0"]]).set_color(colours_1[0]).move_to(C_lagr_vec.get_center())
        grouped_at_zero_value = Group(sol_vec.copy(), A_lagr_vec_at_zero_value.copy(), times.copy(), sol_vec.copy(), B_lagr_vec_at_zero_value.copy(), minus.copy(), sol_vec.copy(), C_lagr_vec_at_zero_value.copy(), equals.copy(), zero.copy()).arrange()
        self.play(*[FadeTransform(combined_vectors_rearranged[s], grouped_at_zero[s]) for s in range(10)])
        self.wait(1.5)
        self.play(*[FadeTransform(grouped_at_zero[s], grouped_at_zero_value[s]) for s in range(10)])
        self.wait(1.5)
        constraint1 = MathTex("\\operatorname{int} = x*x").set_color(colours_1[0]).move_to(3.5*DOWN)
        self.play(FadeIn(constraint1))
        self.wait(1.5)
        self.play(*[FadeTransform(grouped_at_zero_value[s], combined_vectors_rearranged[s]) for s in range(10)], FadeOut(constraint1))
        self.wait(1.5)

        # evaluating the poly at 2 returns us to the second constraints
        A_lagr_vec_at_two = Matrix([["A_1(2)"], ["A_2(2)"], ["A_3(2)"], ["A_4(2)"], ["A_5(2)"], ["A_6(2)"]]).set_color(colours_1[1]).move_to(A_lagr_vec.get_center())
        B_lagr_vec_at_two = Matrix([["B_1(2)"], ["B_2(2)"], ["B_3(2)"], ["B_4(2)"], ["B_5(2)"], ["B_6(2)"]]).set_color(colours_1[1]).move_to(B_lagr_vec.get_center())
        C_lagr_vec_at_two = Matrix([["C_1(2)"], ["C_2(2)"], ["C_3(2)"], ["C_4(2)"], ["C_5(2)"], ["C_6(2)"]]).set_color(colours_1[1]).move_to(C_lagr_vec.get_center())
        grouped_at_two = Group(sol_vec.copy(), A_lagr_vec_at_two.copy(), times.copy(), sol_vec.copy(), B_lagr_vec_at_two.copy(), minus.copy(), sol_vec.copy(), C_lagr_vec_at_two.copy(), equals.copy(), zero.copy()).arrange()
        A_lagr_vec_at_two_value = Matrix([["0"], ["0"], ["1"], ["0"], ["0"], ["0"]]).set_color(colours_1[1]).move_to(A_lagr_vec.get_center())
        B_lagr_vec_at_two_value = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[1]).move_to(B_lagr_vec.get_center())
        C_lagr_vec_at_two_value = Matrix([["0"], ["0"], ["0"], ["1"], ["0"], ["0"]]).set_color(colours_1[1]).move_to(C_lagr_vec.get_center())
        grouped_at_two_value = Group(sol_vec.copy(), A_lagr_vec_at_two_value.copy(), times.copy(), sol_vec.copy(), B_lagr_vec_at_two_value.copy(), minus.copy(), sol_vec.copy(), C_lagr_vec_at_two_value.copy(), equals.copy(), zero.copy()).arrange()
        self.play(*[FadeTransform(combined_vectors_rearranged[s], grouped_at_two[s]) for s in range(10)])
        self.wait(1.5)
        self.play(*[FadeTransform(grouped_at_two[s], grouped_at_two_value[s]) for s in range(10)])
        self.wait(1.5)
        constraint2 = MathTex("\\operatorname{int_2} = \\operatorname{int}*x").set_color(colours_1[1]).move_to(3.5*DOWN)
        self.play(FadeIn(constraint2))
        self.wait(1.5)
        self.play(*[FadeTransform(grouped_at_two_value[s], combined_vectors_rearranged[s]) for s in range(10)], FadeOut(constraint2))
        self.wait(1.5)

class Checking_sol(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        LIGHT = "#00c0f9"
        DARK = "#154bf9"
        colour_special = "#931CFF"
        colour_special_darker = "#9C7900"
        colours_1 = ["#FFCC17", "#FF5555", "#E561E5", "#FF7D54"]
        colours_2 = ["#A479E0", "#4DECAA", "#FFD954", "#22F1E4"]
        colour_lagr = "#0F915B"

        sol_vec = Matrix([[1], ["x"], ["\operatorname{int}"], ["\operatorname{int}_2"], ["\operatorname{int}_3"], ["\operatorname{out}"]]).set_color(colour_special).move_to(5*LEFT)
        A_lagr_vec = Matrix([["A_1(x)"], ["A_2(x)"], ["A_3(x)"], ["A_4(x)"], ["A_5(x)"], ["A_6(x)"]])
        B_lagr_vec = Matrix([["B_1(x)"], ["B_2(x)"], ["B_3(x)"], ["B_4(x)"], ["B_5(x)"], ["B_6(x)"]])
        C_lagr_vec = Matrix([["C_1(x)"], ["C_2(x)"], ["C_3(x)"], ["C_4(x)"], ["C_5(x)"], ["C_6(x)"]])
        times = Tex("*")
        equals = Tex("=")
        minus = Tex("-")
        zero = Tex("0")
        combined_vectors_rearranged = Group(sol_vec.copy(), A_lagr_vec.copy(), times.copy(), sol_vec.copy(), B_lagr_vec.copy(), minus.copy(), sol_vec.copy(), C_lagr_vec.copy(), equals.copy(), zero).arrange()

        # Solution vector stuff
        equals_copy = equals.copy().next_to(combined_vectors_rearranged[1], RIGHT)
        sum_a = MathTex(r'{{1}}{{*A_1(x)}}{{+}}{{x}}{{*A_2(x)}}{{\\+}}{{\operatorname{int}}}{{*A_3(x)}}{{ + }}{{\operatorname{int}_2}}{{*A_4(x)\\ }}{{+ }}{{\operatorname{int}_3}}{{*A_5(x) }}{{+ }}{{\operatorname{out}}}{{*A_6(x)}}').next_to(equals_copy, RIGHT)
        sum_a[0].set_color(colour_special)
        sum_a[3].set_color(colour_special)
        sum_a[6].set_color(colour_special)
        sum_a[10].set_color(colour_special)
        sum_a[13].set_color(colour_special)
        sum_a[16].set_color(colour_special)
        equals_copy2 = equals_copy.copy().next_to(sum_a, RIGHT)
        total_a = MathTex("A(x)").next_to(equals_copy2, RIGHT)
        summing = Group(combined_vectors_rearranged[0], combined_vectors_rearranged[1], equals_copy, sum_a, equals_copy2, total_a)

        # succinct colution vector
        total_b = MathTex("B(x)")
        total_c = MathTex("C(x)")
        hhhh = MathTex("H")
        vanishing_poly = MathTex("Z(x)")
        total_togeth = Group(total_a.copy(), times.copy(), total_b, minus.copy(), total_c, equals.copy(), zero.copy()).arrange().move_to(UP)
        total_togeth_clone = Group(total_a.copy(), times.copy(), total_b.copy(), minus.copy(), total_c.copy(), equals.copy(), zero.copy()).arrange().move_to(3.5*DOWN)

        self.play(FadeIn(combined_vectors_rearranged))
        self.wait(1.5)
        self.play(*[FadeOut(combined_vectors_rearranged[s+2]) for s in range(8)])
        self.wait(1.5)

        # Introduce solution vec again, and remind people how to do dot product
        self.wait(1.5)
        reccys = [0,1,2,3,4,5]
        for s in range(6):
            reccys[s] = Rectangle(width = 3.3, height = 0.6, color = LIGHT).move_to(4.9*LEFT + 2*UP + s*0.8*DOWN)
        self.play(*[FadeIn(reccys[s]) for s in range(6)], FadeIn(summing[2]))
        self.wait(1.5)

        self.play(FadeTransform(reccys[0], sum_a[0]), FadeIn(sum_a[1], sum_a[2]))
        self.wait(1.5)
        self.play(FadeTransform(reccys[1], sum_a[3]), FadeIn(sum_a[3], sum_a[4]))
        self.wait(1.5)
        self.play(FadeTransform(reccys[2], sum_a[6]), FadeTransform(reccys[3], sum_a[10]), FadeTransform(reccys[4], sum_a[13]), FadeTransform(reccys[5], sum_a[16]), *[FadeIn(sum_a[s+5]) for s in range(14)])
        self.wait(1.5)
        self.play(FadeIn(summing[4], summing[5]))
        self.wait(1.5)

        # Now move A(X) to the bottom and bring out rest of the sol vec
        self.play(FadeTransform(summing[5], total_togeth_clone[0]), FadeOut(summing[2], summing[3], summing[4]))
        self.wait(1.5)
        self.play(*[FadeIn(combined_vectors_rearranged[s+2]) for s in range(8)])
        self.wait(1.5)

        # Now introduce B(x) and C(x)
        b_group = Group(combined_vectors_rearranged[3], combined_vectors_rearranged[4])
        c_group = Group(combined_vectors_rearranged[6], combined_vectors_rearranged[7])
        b_rect = SurroundingRectangle(b_group, color = LIGHT)
        c_rect = SurroundingRectangle(c_group, color = LIGHT)
        self.play(FadeTransform(b_rect, total_togeth_clone[2]),
        FadeTransform(c_rect, total_togeth_clone[4]),
        FadeTransform(combined_vectors_rearranged[2].copy(), total_togeth_clone[1]),
        FadeTransform(combined_vectors_rearranged[5].copy(), total_togeth_clone[3]),
        FadeTransform(combined_vectors_rearranged[8].copy(), total_togeth_clone[5]),
        FadeTransform(combined_vectors_rearranged[9].copy(), total_togeth_clone[6]))
        self.wait(1.5)

        # This equation should equal 0
        xrange = MathTex("\\forall x \in [1,4]").next_to(total_togeth_clone[6], RIGHT).shift(1.4*RIGHT)
        self.play(Write (xrange))
        self.wait(1.5)

        # Vector format no longer required
        self.play(*[FadeOut(combined_vectors_rearranged[s]) for s in range(10)])
        self.wait(1.5)
        self.play(*[FadeTransform(total_togeth_clone[s], total_togeth[s]) for s in range(7)], xrange.animate.shift(4.5*UP))
        self.wait(1.5)

        # This is equiv to a multiple of the vanishing poly
        iff = MathTex("a \iff a")
        # Bug prevents above Tex from just being \iff, so add stuff and make it invisible
        iff[0][0].set_color(BLACK)
        iff[0][3].set_color(BLACK)
        equiv_to = Group(total_a.copy(), times.copy(), total_b.copy(), minus.copy(), total_c.copy(), equals.copy(), hhhh, times.copy(), vanishing_poly).arrange().move_to(DOWN) 
        self.play(FadeIn(iff, equiv_to))
        self.wait(1.5)

        # Define the vanishing poly Z(x)
        vanishing_poly_formula = MathTex("(x-1)(x-2)(x-3)(x-4)")
        vanishing_poly_def = Group(vanishing_poly.copy(), equals.copy(), vanishing_poly_formula).arrange().move_to(2*DOWN).set_color("#18E48F")
        self.play(FadeIn(vanishing_poly_def))
        self.wait(1.5)

        # Any poly that is zero on 1 to 4 must be divisble by ..
        any_poly_that_is = Tex(r'Any polynomial that is zero on $x\in[1,4]$ must \\  be divisble by the vanishing polynomial $Z(x)$').move_to(3*DOWN)
        self.play(Write(any_poly_that_is))
        self.wait(1.5)
        self.play(FadeOut(vanishing_poly_def, any_poly_that_is, iff, total_togeth, xrange))
        self.wait(1.5)
        self.play(equiv_to.animate.shift(2*UP))
        self.wait(1.5)

        # equiv to dividing by Z(x) is possible without remainder
        equiv_to_div_by_z = MathTex("{{\\frac{A(x)*B(x)-C(x)}{Z(x)} = }}{{H}}").move_to(1.4*DOWN) 
        self.play(FadeIn(equiv_to_div_by_z, iff))
        self.wait(1.5)
        self.play(Indicate(equiv_to_div_by_z[1]))
        self.wait(1.5)





class Ending(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        LIGHT = "#00c0f9"
        DARK = "#154bf9"
        colour_special = "#931CFF"
        colour_special_darker = "#9C7900"
        colours_1 = ["#FFCC17", "#FF5555", "#E561E5", "#FF7D54"]
        colours_2 = ["#A479E0", "#4DECAA", "#FFD954", "#22F1E4"]
        colour_lagr = "#0F915B"

        lagrange_axes = Axes(x_range=[0,8],y_range=[0,6], y_length=5, tips=False, axis_config={"include_numbers": True}).shift(0.8*UP)
        vec = [[1, 3, 5, 1]]
        vecc = Matrix(vec).shift(3.5*DOWN + 2.3*LEFT)
        points = [Dot(lagrange_axes.coords_to_point(s+1,vec[0][s])) for s in range(4)]
        lagrange_bases1 = lambda x: langrage_basis(x, 0)
        lagrange_bases2 = lambda x: langrage_basis(x, 1)
        lagrange_bases3 = lambda x: langrage_basis(x, 2)
        lagrange_bases4 = lambda x: langrage_basis(x, 3)
        lagrangey = [lagrange_bases1, lagrange_bases2, lagrange_bases3, lagrange_bases4]
        plots = [*[lagrange_axes.plot(lagrangey[i], color = colours_2[i]) for i in range(4)]]
        lagr_equations = [0,1,2,3]
        lagr_succinct = [0,1,2,3]
        lagr_equations[0] = MathTex("-\\frac{1}{6}x^3+\\frac{3}{2}x^2-\\frac{13}{3}x+4").set_color(colours_2[0])
        lagr_equations[1] = MathTex("\\frac{1}{2}x^3 - 4x^2 + \\frac{19}{2}x - 6").set_color(colours_2[1])
        lagr_equations[2] = MathTex("-\\frac{1}{2}x^3 + \\frac{7}{2}x^2 - 7x + 4").set_color(colours_2[2])
        lagr_equations[3] = MathTex("-\\frac{1}{6}x^3+\\frac{3}{2}x^2-\\frac{13}{3}x+4").set_color(colours_2[3])
        lagr_succinct[0] = MathTex("L_1(x)").set_color(colours_2[0]).move_to(4*RIGHT)
        lagr_succinct[1] = MathTex("L_2(x)").set_color(colours_2[1]).move_to(4*RIGHT)
        lagr_succinct[2] = MathTex("L_3(x)").set_color(colours_2[2]).move_to(4*RIGHT)
        lagr_succinct[3] = MathTex("L_4(x)").set_color(colours_2[3]).move_to(4*RIGHT)
        lagr_equations_with_succinct = [0,1,2,3]
        for s in range(4):
            lagr_equations_with_succinct[s] = Group(lagr_equations[s], lagr_succinct[s]).arrange(DOWN).shift(4*RIGHT)
        multiply_to_fit = [0,1]
        multiply_to_fit[0] = MathTex("3*(\\frac{1}{2}x^3 - 4x^2 + \\frac{19}{2}x - 6)").set_color(colours_2[1])
        multiply_to_fit[1] = MathTex("5*(-\\frac{1}{2}x^3 + \\frac{7}{2}x^2 - 7x + 4)").set_color(colours_2[2])
        multiply_to_fit_succinct = [0,1]
        multiply_to_fit_succinct[0] = Group(multiply_to_fit[0], MathTex("3*L_2(x)").set_color(colours_2[1])).arrange(DOWN).shift(4*RIGHT)
        multiply_to_fit_succinct[1] = Group(multiply_to_fit[1], MathTex("5*L_3(x)").set_color(colours_2[2])).arrange(DOWN).shift(4*RIGHT)
        sum_of_lagr_for_interp_example = Group(lagr_succinct[0].copy(), MathTex("+3*L_2(x)").set_color(colours_2[1]), MathTex("+5*L_3(x)").set_color(colours_2[2]), MathTex("+L_4(x)").set_color(colours_2[3])).arrange(DOWN).shift(4*RIGHT)

        interpoly = lambda x: vec[0][0]*langrage_basis(x, 0)+vec[0][1]*langrage_basis(x, 1)+vec[0][2]*langrage_basis(x, 2)+vec[0][3]*langrage_basis(x, 3)
        interpoly_plot = lagrange_axes.plot(interpoly, color = BLUE)
        interpoly_equ = MathTex("-x^3 + 6x^2 - 9x + 5").set_color(BLUE).move_to(sum_of_lagr_for_interp_example.get_center())
        interpo_point = Dot(lagrange_axes.coords_to_point(0,5), color = RED)
        vecc_basis_2 = lambda x: vec[0][1]*langrage_basis(x, 1)
        vecc_basis_3 = lambda x: vec[0][2]*langrage_basis(x, 2)
        vecc_plots = [lagrange_axes.plot(vecc_basis_2, color = colours_2[1]), lagrange_axes.plot(vecc_basis_3, color = colours_2[2])]

        AAA = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[0])
        BBB = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[0])
        CCC = Matrix([["0"], ["0"], ["1"], ["0"], ["0"], ["0"]]).set_color(colours_1[0])
        AAA2 = Matrix([["0"], ["0"], ["1"], ["0"], ["0"], ["0"]]).set_color(colours_1[1])
        BBB2 = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[1])
        CCC2 = Matrix([["0"], ["0"], ["0"], ["1"], ["0"], ["0"]]).set_color(colours_1[1])
        AAA3 = Matrix([["0"], ["1"], ["0"], ["1"], ["0"], ["0"]]).set_color(colours_1[2])
        BBB3 = Matrix([["1"], ["0"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[2])
        CCC3 = Matrix([["0"], ["0"], ["0"], ["0"], ["1"], ["0"]]).set_color(colours_1[2])
        AAA4 = Matrix([["5"], ["0"], ["0"], ["0"], ["1"], ["0"]]).set_color(colours_1[3])
        BBB4 = Matrix([["1"], ["0"], ["0"], ["0"], ["0"], ["0"]]).set_color(colours_1[3])
        CCC4 = Matrix([["0"], ["0"], ["0"], ["0"], ["0"], ["1"]]).set_color(colours_1[3])

        vec2 = [[0, 0, 0, 5]]
        vecc2 = Matrix(vec2).set_column_colors(colours_1[0], colours_1[1], colours_1[2], colours_1[3]).shift(3.5*DOWN + 2.3*LEFT)
        firs_a_vec_function = lambda x: vec2[0][3]*langrage_basis(x, 3)
        first_a_vec = lagrange_axes.plot(firs_a_vec_function, color = GREEN)
        a_vec = Group(AAA.copy(), AAA2.copy(), AAA3.copy(), AAA4.copy()).arrange().move_to(3.5*RIGHT)
        A1_lagr = MathTex("\\frac{5}{6}x^3 - 5x^2 + \\frac{55}{6}x - 5").shift(3.5*RIGHT).set_color(GREEN)

        constraint1 = Group(AAA, BBB, CCC).arrange().scale(0.7)
        constraint2 = Group(AAA2, BBB2, CCC2).arrange().scale(0.7)
        constraint3 = Group(AAA3, BBB3, CCC3).arrange().scale(0.7)
        constraint4 = Group(AAA4, BBB4, CCC4).arrange().scale(0.7)
        constraints_real = Group(constraint1, constraint2, constraint3, constraint4).arrange_in_grid().move_to(3*RIGHT)
        copies = Group(AAA.copy(), AAA2.copy(), AAA3.copy(), AAA4.copy())

        steps = [0,1,2,3]
        steps[0] = MathTex("\operatorname{int} = x*x").set_color(colours_1[0]).move_to(constraints_real[0].get_center())
        steps[1] = MathTex("\operatorname{int}_2 = \operatorname{int}*x").set_color(colours_1[1]).move_to(constraints_real[1].get_center())
        steps[2] = MathTex("\operatorname{int}_3 = \operatorname{int}_2 + x").set_color(colours_1[2]).move_to(constraints_real[2].get_center())
        steps[3] = MathTex("\operatorname{out} = \operatorname{int}_3 + 5").set_color(colours_1[3]).move_to(constraints_real[3].get_center())
        steps_group = Group(steps[0], steps[1], steps[2], steps[3]).arrange(DOWN).move_to(3*RIGHT)
        equation = MathTex("x^{3} + x + 5 = 35", color=colour_special)

        A2_lagr = MathTex("-\\frac{2}{3}x^3 + 5x^2 + -\\frac{34}{3}x + 8").scale(0.6)
        A3_lagr = MathTex("\\frac{1}{2}x^3 - 4x^2 + \\frac{19}{2}x - 6").scale(0.6)
        A4_lagr = MathTex("-x^3 + 7x^2 + -14x + 8").scale(0.6)
        A5_lagr = MathTex("\\frac{1}{6}x^3 - x^2 + \\frac{11}{6}x - 1").scale(0.6)
        A6_lagr = MathTex("0x^3 + 0x^2 + 0x + 0").scale(0.6)
        A = [A1_lagr, A2_lagr, A3_lagr, A4_lagr, A5_lagr, A6_lagr]
        for s in range(5):
            A[s+1].move_to(2.5*LEFT + 2*UP + 0.8*(s+1)*DOWN).set_color(GREEN)
        A_interp_succinct = [0,1,2,3,4,5]
        A_interp_succinct[0] = MathTex("A_{1}(x)").scale(0.7).move_to(A[1].get_center() + 0.8*UP).set_color(BLUE)
        A_interp_succinct[1] = MathTex("A_{2}(x)").scale(0.7).move_to(A[1].get_center()).set_color(BLUE)
        A_interp_succinct[2] = MathTex("A_{3}(x)").scale(0.7).move_to(A[2].get_center()).set_color(BLUE)
        A_interp_succinct[3] = MathTex("A_{4}(x)").scale(0.7).move_to(A[3].get_center()).set_color(BLUE)
        A_interp_succinct[4] = MathTex("A_{5}(x)").scale(0.7).move_to(A[4].get_center()).set_color(BLUE)
        A_interp_succinct[5] = MathTex("A_{6}(x)").scale(0.7).move_to(A[5].get_center()).set_color(BLUE)
        B_interp_succinct = [0,1,2,3,4,5]
        B_interp_succinct[0] = MathTex("B_{1}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct[1] = MathTex("B_{2}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct[2] = MathTex("B_{3}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct[3] = MathTex("B_{4}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct[4] = MathTex("B_{5}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct[5] = MathTex("B_{6}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct = [0,1,2,3,4,5]
        C_interp_succinct[0] = MathTex("C_{1}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct[1] = MathTex("C_{2}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct[2] = MathTex("C_{3}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct[3] = MathTex("C_{4}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct[4] = MathTex("C_{5}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct[5] = MathTex("C_{6}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct_group = Group(B_interp_succinct[0], B_interp_succinct[1], B_interp_succinct[2], B_interp_succinct[3], B_interp_succinct[4], B_interp_succinct[5]).scale(0.7).arrange(DOWN, buff=0.33).move_to(2*RIGHT)
        C_interp_succinct_group = Group(C_interp_succinct[0], C_interp_succinct[1], C_interp_succinct[2], C_interp_succinct[3], C_interp_succinct[4], C_interp_succinct[5]).scale(0.7).arrange(DOWN, buff=0.32).move_to(6.3*RIGHT)

        L_polys = [0,1,2,3]
        L_polys[0] = MathTex("L_{1}(x)").scale(0.7).move_to(a_vec[0].get_top() + 0.6*UP).set_color(colour_lagr)
        L_polys[1] = MathTex("L_{2}(x)").scale(0.7).move_to(a_vec[1].get_top() + 0.6*UP).set_color(colour_lagr)
        L_polys[2] = MathTex("L_{3}(x)").scale(0.7).move_to(a_vec[2].get_top() + 0.6*UP).set_color(colour_lagr)
        L_polys[3] = MathTex("L_{4}(x)").scale(0.7).move_to(a_vec[3].get_top() + 0.6*UP).set_color(colour_lagr)

        sol_vec = Matrix([[1], ["x"], ["\operatorname{int}"], ["\operatorname{int}_2"], ["\operatorname{int}_3"], ["\operatorname{out}"]]).set_color(colour_special).move_to(5*LEFT)
        A_lagr_vec = Matrix([["A_1(x)"], ["A_2(x)"], ["A_3(x)"], ["A_4(x)"], ["A_5(x)"], ["A_6(x)"]])
        B_lagr_vec = Matrix([["B_1(x)"], ["B_2(x)"], ["B_3(x)"], ["B_4(x)"], ["B_5(x)"], ["B_6(x)"]])
        C_lagr_vec = Matrix([["C_1(x)"], ["C_2(x)"], ["C_3(x)"], ["C_4(x)"], ["C_5(x)"], ["C_6(x)"]])
        times = Tex("*")
        equals = Tex("=")
        minus = Tex("-")
        zero = Tex("0")
        combined_vectors_rearranged = Group(sol_vec.copy(), A_lagr_vec.copy(), times.copy(), sol_vec.copy(), B_lagr_vec.copy(), minus.copy(), sol_vec.copy(), C_lagr_vec.copy(), equals.copy(), zero).arrange()

        # Solution vector stuff
        equals_copy = equals.copy().next_to(combined_vectors_rearranged[1], RIGHT)
        sum_a = MathTex(r'{{1}}{{*A_1(x)}}{{+}}{{x}}{{*A_2(x)}}{{\\+}}{{\operatorname{int}}}{{*A_3(x)}}{{ + }}{{\operatorname{int}_2}}{{*A_4(x)\\ }}{{+ }}{{\operatorname{int}_3}}{{*A_5(x) }}{{+ }}{{\operatorname{out}}}{{*A_6(x)}}').next_to(equals_copy, RIGHT)
        sum_a[0].set_color(colour_special)
        sum_a[3].set_color(colour_special)
        sum_a[6].set_color(colour_special)
        sum_a[10].set_color(colour_special)
        sum_a[13].set_color(colour_special)
        sum_a[16].set_color(colour_special)
        equals_copy2 = equals_copy.copy().next_to(sum_a, RIGHT)
        total_a = MathTex("A(x)").next_to(equals_copy2, RIGHT)
        summing = Group(combined_vectors_rearranged[0], combined_vectors_rearranged[1], equals_copy, sum_a, equals_copy2, total_a)

        # succinct colution vector
        total_b = MathTex("B(x)")
        total_c = MathTex("C(x)")
        hhhh = MathTex("H")
        vanishing_poly = MathTex("Z(x)")
        total_togeth = Group(total_a.copy(), times.copy(), total_b, minus.copy(), total_c, equals.copy(), zero.copy()).arrange().move_to(UP)
        total_togeth_clone = Group(total_a.copy(), times.copy(), total_b.copy(), minus.copy(), total_c.copy(), equals.copy(), zero.copy()).arrange().move_to(3.5*DOWN)
        

        surrounder = [0,1,2,3]
        surrounder[0] = SurroundingRectangle(constraints_real[0], color = colours_1[0])
        surrounder[1] = SurroundingRectangle(constraints_real[1], color = colours_1[1])
        surrounder[2] = SurroundingRectangle(constraints_real[2], color = colours_1[2])
        surrounder[3] = SurroundingRectangle(constraints_real[3], color = colours_1[3])
        final_surrounder = SurroundingRectangle(steps_group, color = colour_special)
        self.play(Write(equation.move_to(steps_group.get_center() + 3*UP)))
        self.wait(0.7) 
        self.play(FadeTransform(equation.copy(), final_surrounder), FadeIn(steps_group))
        # self.play(*[Write(steps_group[s][0] for s in range(3))])
        sol_vec = Matrix([[1], ["x"], ["\operatorname{int}"], ["\operatorname{int}_2"], ["\operatorname{int}_3"], ["\operatorname{out}"]]).set_color(colour_special).move_to(5*LEFT)
        self.play(FadeOut(final_surrounder))
        self.wait(0.7)
        self.play(*[FadeTransform(steps_group[s], surrounder[s]) for s in range(4)], *[FadeIn(constraints_real[s]) for s in range(4)], FadeOut(equation), FadeIn(sol_vec))
        self.play(*[FadeOut(surrounder[s]) for s in range(4)])
        self.wait(0.7)
        a_vecs = Group(constraints_real[0][0].copy(), constraints_real[1][0].copy(), constraints_real[2][0].copy(), constraints_real[3][0].copy()).arrange(buff = 0.1)
        b_vecs = Group(constraints_real[0][1].copy(), constraints_real[1][1].copy(), constraints_real[2][1].copy(), constraints_real[3][1].copy()).arrange(buff = 0.1)
        c_vecs = Group(constraints_real[0][2].copy(), constraints_real[1][2].copy(), constraints_real[2][2].copy(), constraints_real[3][2].copy()).arrange(buff = 0.1)
        vec_groups = Group(a_vecs, b_vecs, c_vecs).arrange(buff = 1.3)
        L_basis_group = Group(L_polys[0], L_polys[1], L_polys[2], L_polys[3]).scale(0.7).arrange(buff = 0.18).move_to(a_vecs.get_center() + 2*UP)
        L_basis_group2 = L_basis_group.copy().move_to(b_vecs.get_center() + 2*UP)
        L_basis_group3 = L_basis_group.copy().move_to(c_vecs.get_center() + 2*UP)
        self.play(*[FadeTransform(constraints_real[s][0], a_vecs[s]) for s in range(4)], *[FadeTransform(constraints_real[s][1], b_vecs[s]) for s in range(4)], *[FadeTransform(constraints_real[s][2], c_vecs[s]) for s in range(4)], FadeOut(sol_vec))
        self.wait(0.7)

        reccys = [0,1,2,3,4,5]
        for s in range(6):
            reccys[s] = Rectangle(width = 2.8, height = 0.4, color = LIGHT).move_to(a_vecs.get_center() + 1.4*UP + s*0.56*DOWN)
        reccys2 = [0,1,2,3,4,5]
        for s in range(6):
            reccys2[s] = Rectangle(width = 2.8, height = 0.4, color = LIGHT).move_to(b_vecs.get_center() + 1.4*UP + s*0.56*DOWN)
        reccys3 = [0,1,2,3,4,5]
        for s in range(6):
            reccys3[s] = Rectangle(width = 2.8, height = 0.4, color = LIGHT).move_to(c_vecs.get_center() + 1.4*UP + s*0.56*DOWN)
        A_interp_succinct_group = Group(A_interp_succinct[0], A_interp_succinct[1], A_interp_succinct[2], A_interp_succinct[3], A_interp_succinct[4], A_interp_succinct[5]).scale(0.7).arrange(DOWN, buff=0.33).move_to(2.4*LEFT)
        

        # Display Lagrange bases above the b and c vec groups
        L_basis_group2 = L_basis_group.copy().move_to(b_vecs.get_center() + 2*UP)
        L_basis_group3 = L_basis_group.copy().move_to(c_vecs.get_center() + 2*UP)
        self.play(*[Write(L_basis_group[s]) for s in range(4)], *[Write(L_basis_group2[s]) for s in range(4)], *[Write(L_basis_group3[s]) for s in range(4)])
        self.wait(0.7)

        # Rectangles around b_vec and c_vec entries
        self.play(*[FadeIn(reccys[s]) for s in range(6)], *[FadeIn(reccys2[s]) for s in range(6)], *[FadeIn(reccys3[s]) for s in range(6)])
        self.wait(0.7)
        self.play(*[ReplacementTransform(reccys[s], A_interp_succinct_group[s]) for s in range(6)], *[ReplacementTransform(reccys2[s], B_interp_succinct_group[s]) for s in range(6)], *[ReplacementTransform(reccys3[s], C_interp_succinct_group[s]) for s in range(6)])
        self.wait(0.7)

        # Fade out constraint vectors and Lagrange bases
        self.play(FadeOut(L_basis_group, L_basis_group2, L_basis_group3, a_vecs, b_vecs, c_vecs))
        self.wait(0.7)

        # Make lagr interp into actual vectors rather than just grouping, and bring in sol vec again
        A_lagr_vec = Matrix([["A_1(x)"], ["A_2(x)"], ["A_3(x)"], ["A_4(x)"], ["A_5(x)"], ["A_6(x)"]])
        B_lagr_vec = Matrix([["B_1(x)"], ["B_2(x)"], ["B_3(x)"], ["B_4(x)"], ["B_5(x)"], ["B_6(x)"]])
        C_lagr_vec = Matrix([["C_1(x)"], ["C_2(x)"], ["C_3(x)"], ["C_4(x)"], ["C_5(x)"], ["C_6(x)"]])
        times = Tex("*")
        equals = Tex("=")
        minus = Tex("-")
        zero = Tex("0")
        combined_vectors = Group(sol_vec, A_lagr_vec, times, sol_vec.copy(), B_lagr_vec, equals, sol_vec.copy(), C_lagr_vec).arrange()
        combined_vectors_rearranged = Group(sol_vec.copy(), A_lagr_vec.copy(), times.copy(), sol_vec.copy(), B_lagr_vec.copy(), minus.copy(), sol_vec.copy(), C_lagr_vec.copy(), equals.copy(), zero).arrange()
        self.play(*[FadeTransform(A_interp_succinct_group[s], A_lagr_vec[0][s]) for s in range(6)], *[FadeTransform(B_interp_succinct_group[s], B_lagr_vec[0][s]) for s in range(6)], *[FadeTransform(C_interp_succinct_group[s], C_lagr_vec[0][s]) for s in range(6)], FadeIn(A_lagr_vec[1], A_lagr_vec[2], B_lagr_vec[1], B_lagr_vec[2], C_lagr_vec[1], C_lagr_vec[2]))
        self.wait(0.7)
        self.play(FadeIn(combined_vectors[0], combined_vectors[2], combined_vectors[3], combined_vectors[5], combined_vectors[6]))
        self.wait(0.7)
        self.play(*[FadeTransform(combined_vectors[s], combined_vectors_rearranged[s]) for s in range(8)], FadeIn(combined_vectors_rearranged[8], combined_vectors_rearranged[9]))
        self.wait(0.7)


        # This equation should equal 0
        xrange = MathTex("\\forall x \in [1,4]").next_to(total_togeth_clone[6], RIGHT).shift(1.4*RIGHT)
        self.play(*[Write(total_togeth_clone[s]) for s in range(7)], Write(xrange))
        self.wait(0.7)

        # Vector format no longer required
        self.play(*[FadeOut(combined_vectors_rearranged[s]) for s in range(10)])
        self.wait(0.7)
        self.play(*[FadeTransform(total_togeth_clone[s], total_togeth[s]) for s in range(7)], xrange.animate.shift(4.5*UP))
        self.wait(0.7)

        # This is equiv to a multiple of the vanishing poly
        iff = MathTex("a \iff a")
        # Bug prevents above Tex from just being \iff, so add stuff and make it invisible
        iff[0][0].set_color(BLACK)
        iff[0][3].set_color(BLACK)
        equiv_to = Group(total_a.copy(), times.copy(), total_b.copy(), minus.copy(), total_c.copy(), equals.copy(), hhhh, times.copy(), vanishing_poly).arrange().move_to(DOWN) 
        self.play(FadeIn(iff, equiv_to))
        self.wait(0.7)

        # Define the vanishing poly Z(x)
        vanishing_poly_formula = MathTex("(x-1)(x-2)(x-3)(x-4)")
        vanishing_poly_def = Group(vanishing_poly.copy(), equals.copy(), vanishing_poly_formula).arrange().move_to(2*DOWN).set_color("#18E48F")
        self.play(FadeIn(vanishing_poly_def))
        self.wait(0.7)

        # Any poly that is zero on 1 to 4 must be divisble by ..
        self.play(FadeOut(vanishing_poly_def, iff, total_togeth, xrange))
        self.wait(0.7)
        self.play(equiv_to.animate.shift(2*UP))
        self.wait(0.7)

        # equiv to dividing by Z(x) is possible without remainder
        equiv_to_div_by_z = MathTex("{{\\frac{A(x)*B(x)-C(x)}{Z(x)} = }}{{H}}").move_to(1.4*DOWN) 
        self.play(FadeIn(equiv_to_div_by_z, iff))
        self.wait(0.7)
        self.play(Indicate(equiv_to_div_by_z[1]))
        self.wait(0.7)

        self.play(FadeOut(equiv_to_div_by_z, iff, equiv_to))
        self.wait(0.6)
        in_collab_with = Tex("In collaboration with ZK-Garage")
        special_thanks = Tex(r'Special thanks to \\ Todd Norton')
        zkgarage = ImageMobject("Images/zkgarage.jpg").scale(0.5)
        zgroup = Group(in_collab_with, zkgarage).arrange(DOWN).move_to(UP)
        self.play(Write(in_collab_with))
        self.play(FadeIn(zkgarage))
        self.wait(1.5)
        self.play(Write(special_thanks.move_to(DOWN)))
        self.wait(1.5)



# Created by animator to see how a 3d scne would work
class Cuboid(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes, Dodecahedron(graph_config={"vertex_config": {"fill_opacity": 0, "stroke_opacity": 0}}))

class Rectangles(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(x_range=(0, 6, 1), y_range=(0, 5, 1), z_range=(0, 4, 1)).scale(0.6).move_to(ORIGIN)
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        dots = VGroup()

        # Sphere dots
        # x = 0
        # y = 0
        # z = 0

        # while x <= 2:
        #     while y <= 5:
        #         while z <= 3:
        #             dots.add(Dot3D(point=axes.c2p(x, y, z), radius=0.1, color=BLUE, resolution=(16, 16)))
        #             z += 1
        #         z = 0
        #         y += 1
        #     y = 0
        #     # 0.8 for 5 dots
        #     x += 1
        # self.add(axes, dots)

        x = 0
        y = 0

        while x <= 2:
            while y <= 5:
                dots.add(Dot(point=axes.c2p(x, y, 0), radius=0.2, color=BLUE))
                y += 1
            y = 0
            x += 1

        dots2 = dots.copy().set_color(YELLOW).set_z(axes.c2p(0, 0, 1)[2])
        dots3 = dots.copy().set_color(PINK).set_z(axes.c2p(0, 0, 2)[2])
        dots4 = dots.copy().set_color(WHITE).set_z(axes.c2p(0, 0, 3)[2])

        self.add(axes, dots, dots2, dots3, dots4)





# Created for ZKGarage, trying to create an image for the twitter background but more generally too
class PlonkEqu(Scene):
    def construct(self):
        self.camera.background_color = BLACK


        colour_special = "#931CFF"
        colour_special_darker = "#9C7900"
        colours_1 = ["#FFCC17", "#FF5555", "#E561E5", "#FF7D54"]
        colours_2 = ["#A479E0", "#4DECAA", "#FFD954", "#22F1E4"]
        colour_lagr = "#0F915B"

        colour_plonk = "#9b49a6"
        colours = ["#3ca6d6", "#dc8724", "#34d140"]

        plonk_equ = MathTex("{{a(x)}}{{q_L(x)}}{{+}}{{b(x)}}{{q_R(x)}}{{+}}{{c(x)}}{{q_O(x)}}{{+}}{{a(x)}}{{b(x)}}{{q_M(x)}}{{+}}{{q_C(x)}}{{= 0}}").scale(0.8).shift(UP)
        # a(x) colour
        plonk_equ[0].set_color(colours_2[1])
        plonk_equ[9].set_color(colours_2[1])
        # b(x) colour
        plonk_equ[3].set_color(colours_2[2])
        plonk_equ[10].set_color(colours_2[2])
        # c(x) colour
        plonk_equ[6].set_color(colours_2[3])
        # selector poly colour
        plonk_equ[1].set_color(colours_2[0])
        plonk_equ[4].set_color(colours_2[0])
        plonk_equ[7].set_color(colours_2[0])
        plonk_equ[11].set_color(colours_2[0])
        plonk_equ[13].set_color(colours_2[0])

        self.play(FadeIn(plonk_equ))
        print(len(plonk_equ))
        self.wait(1.5)
        circuit = circuit_pattern().scale(0.7).shift(DOWN)
        self.play(FadeIn(circuit))
        # self.circuit_pattern()
        self.wait(1.5)



def circuit_pattern():
    result = VGroup()


    colours_2 = ["#A479E0", "#4DECAA", "#FFD954", "#22F1E4"]
    colour_plonk = "#9b49a6"
    colours = ["#3ca6d6", "#dc8724", "#34d140"]

    unvisibleline1=Line(color=BLACK).shift(UP)
    unvisibleline2=Line(color=BLACK).shift(DOWN*1.5)

    circle = Circle(color=colours_2[0]).scale(0.5)

    linetop1=Line(unvisibleline1.get_start(),circle, color = colours_2[1])

    linetop2=Line(unvisibleline1.get_end(),circle, color = colours_2[2])

    linebottom1=Line(circle,unvisibleline2.get_center(), color = colours_2[3])

    linetop1_label=Tex("a").move_to(linetop1,UL).shift(UP*0.5).shift(LEFT*0.5).set_color(colours_2[1])
    linetop2_label=Tex("b").move_to(linetop2,UR).shift(RIGHT*0.6).shift(UP*0.5).set_color(colours_2[2])
    linebottom1_label=Tex("c").move_to(linebottom1,DR).shift(RIGHT*0.6).shift(DOWN*0.5).set_color(colours_2[3])

    result.add(circle,linetop1,linetop2,linebottom1,linetop1_label,linetop2_label,linebottom1_label,unvisibleline1,unvisibleline2)
    return result