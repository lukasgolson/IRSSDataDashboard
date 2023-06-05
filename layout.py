# Code adopted from https://discuss.streamlit.io/t/st-footer/6447/15

import streamlit as st
from htbuilder import HtmlElement, div, hr, a, p, styles
from streamlit.components.v1 import html


def link(href: str, text: str, **style) -> a:
    """Create an HTML element with the given href, text and style."""
    return a(_href=href, _target="_blank", style=styles(**style))(text)


def get_styles() -> str:
    """Return the CSS styles for the layout."""
    with open('layout/footer.css', 'r') as f:
        return "<style>\n" + f.read() + "\n</style>"


def get_js() -> str:
    """Return the JS for the layout."""
    with open('layout/footer.js', 'r') as f:
        return "<script>\n" + f.read() + "\n</script>"


def inject_footer(*args):
    """Add the layout to the Streamlit app with the given HTML elements or strings."""
    body = p(id='newFooter')
    foot = div(id='styledDiv')(hr(id='styledHr'), body)

    for arg in args:
        body(arg) if isinstance(arg, (str, HtmlElement)) else None

    st.markdown(get_styles(), unsafe_allow_html=True)
    st.markdown(str(foot), unsafe_allow_html=True)
    html(get_js(), height=0)


def set_page_config():
    """Set the Streamlit page configuration."""
    st.set_page_config(page_title="Java Jotter Dashboard", page_icon="☕", layout="centered",
                       initial_sidebar_state="expanded", menu_items={
            'Get Help': 'https://github.com/lukasdragon/IRSSDataDashboard',
            'Report a bug': "https://github.com/lukasdragon/IRSSDataDashboard/issues",
        })


def footer():
    """Add the footer to the Streamlit app."""
    args = [
        "Dashboard made using Streamlit; Postgres hosting with Supabase; Scrapper written in C# -- all with ❤️ "
        "by ", link("https://github.com/lukasdragon", "Lukas Olson"),
    ]
    inject_footer(*args)


def apply_layout():
    """Apply the layout and footer to the Streamlit app."""
    set_page_config()
    footer()


if __name__ == "__main__":
    apply_layout()
